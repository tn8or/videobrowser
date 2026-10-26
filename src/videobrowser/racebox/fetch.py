"""Download RaceBox ``.vbo`` sessions from racebox.pro with Playwright.

Needs the ``fetch`` extra (``pip install -e ".[fetch]"`` and
``playwright install chromium``) and a saved login from
``videobrowser racebox-login``.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import sqlite3
import sys
import time
from pathlib import Path

LOGIN_URL = "https://www.racebox.pro/webapp/login"
SESSIONS_URL = "https://www.racebox.pro/webapp/sessions"


class DownloadTracker:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS downloaded_sessions (
                    session_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

    def is_downloaded(self, session_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM downloaded_sessions WHERE session_id = ?", (session_id,)
            )
            return cursor.fetchone() is not None

    def mark_downloaded(self, session_id: str, filename: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO downloaded_sessions (session_id, filename) VALUES (?, ?)",
                (session_id, filename),
            )
            conn.commit()


def extract_session_id(url: str) -> str:
    """Extract session ID from URL like /webapp/sessions/12345"""
    match = re.search(r"/sessions/(\d+)", url)
    return match.group(1) if match else url


# RaceBox put a Cloudflare Turnstile challenge on the login form, which an
# automated browser can't pass. We log in once interactively
# (``videobrowser racebox-login``), persist the cookies here, and reuse them
# for headless runs.
CONFIG_DIR = Path.home() / ".videobrowser"
AUTH_STATE_PATH = CONFIG_DIR / "racebox_auth.json"
CHROME_PROFILE_DIR = CONFIG_DIR / "racebox_chrome_profile"
# Where the login lived before the fetcher moved into the package.
LEGACY_AUTH_STATE_PATH = (
    Path(__file__).resolve().parents[3] / "racebox" / "fetcher" / "auth_state.json"
)


def migrate_legacy_auth(
    auth_state: Path | None = None, legacy: Path | None = None
) -> None:
    """Copy a login saved under ``racebox/fetcher/`` to its new place, once."""
    auth_state = auth_state or AUTH_STATE_PATH
    legacy = legacy or LEGACY_AUTH_STATE_PATH
    if auth_state.exists() or not legacy.is_file():
        return
    auth_state.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(legacy, auth_state)
    auth_state.chmod(0o600)
    print(f"racebox: copied saved login from {legacy} to {auth_state}")


class NotLoggedInError(RuntimeError):
    pass


def is_logged_in(page) -> bool:
    page.goto(SESSIONS_URL, wait_until="domcontentloaded")
    return "/webapp/login" not in page.url


def new_context(browser, auth_state: Path | None = None):
    auth_state = auth_state or AUTH_STATE_PATH
    if auth_state.exists():
        return browser.new_context(
            accept_downloads=True, storage_state=str(auth_state)
        )
    return browser.new_context(accept_downloads=True)


def ensure_logged_in(page) -> None:
    """Raise unless the saved cookies still give a logged-in session."""
    if is_logged_in(page):
        return
    raise NotLoggedInError(
        "not logged in to racebox.pro (login is behind a Cloudflare captcha). "
        "Run once: videobrowser racebox-login"
    )


def find_chrome() -> str:
    """Path to a desktop Chrome/Chromium; CHROME_BIN overrides the search."""
    override = os.environ.get("CHROME_BIN")
    if override:
        if not Path(override).is_file():
            raise FileNotFoundError(f"CHROME_BIN does not point to a file: {override}")
        return override
    candidates: list[str] = []
    if sys.platform == "darwin":
        candidates.append("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    elif os.name == "nt":
        for base in (
            os.environ.get("PROGRAMFILES"),
            os.environ.get("PROGRAMFILES(X86)"),
            os.environ.get("LOCALAPPDATA"),
        ):
            if base:
                candidates.append(str(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe"))
    for path in candidates:
        if Path(path).exists():
            return path
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(
        "could not find Google Chrome; install it or set CHROME_BIN to its path"
    )


CDP_PORT = 9333


def _tty_state():
    try:
        import termios

        return termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        return None


def _restore_tty(state) -> None:
    if state is None:
        return
    try:
        import termios

        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, state)
    except Exception:
        pass


def interactive_login(auth_state: Path | None = None) -> None:
    """Log in by hand in a plain (non-automated) Chrome, then copy its cookies.

    Turnstile rejects Playwright-launched browsers even when a human clicks, so
    Chrome is started as a normal process with its own profile; we only attach
    over CDP after the user confirms they are logged in.
    """
    import subprocess
    import urllib.request

    from playwright.sync_api import sync_playwright

    auth_state = auth_state or AUTH_STATE_PATH
    chrome = find_chrome()
    CHROME_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    tty = _tty_state()
    if os.name == "nt":
        detach = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        detach = {"start_new_session": True}
    proc = subprocess.Popen(
        [
            chrome,
            f"--user-data-dir={CHROME_PROFILE_DIR}",
            f"--remote-debugging-port={CDP_PORT}",
            "--no-first-run",
            "--no-default-browser-check",
            LOGIN_URL,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **detach,
    )
    try:
        # Chrome can leave the terminal in raw mode (Enter shows as ^M).
        time.sleep(1.0)
        _restore_tty(tty)
        input(
            "Log in to RaceBox in the Chrome window (solve the captcha), "
            "then press Enter here… "
        )
        for _ in range(20):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/version", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{CDP_PORT}")
            context = browser.contexts[0]
            page = context.new_page()
            if not is_logged_in(page):
                raise NotLoggedInError("still not logged in; run videobrowser racebox-login again")
            auth_state.parent.mkdir(parents=True, exist_ok=True)
            context.storage_state(path=str(auth_state))
            auth_state.chmod(0o600)
            print(f"Saved login session to {auth_state}")
    finally:
        _restore_tty(tty)
        # Chrome runs in its own session: stop the whole group, the terminal
        # closing no longer does.
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def load_all_sessions(page, max_scrolls: int = 20) -> None:
    page.goto(SESSIONS_URL, wait_until="domcontentloaded")
    accept = page.locator("button:has-text('I ACCEPT'), button:has-text('Accept')")
    if accept.count() > 0:
        try:
            accept.first.click(timeout=3000)
            time.sleep(0.5)
        except Exception:
            pass

    last_count = -1
    for _ in range(max_scrolls):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1.5)
        load_more = page.locator(
            "button:has-text('Load more'), button:has-text('Show more')"
        )
        if load_more.count() > 0 and load_more.first.is_visible():
            try:
                load_more.first.click()
                time.sleep(1.5)
            except Exception:
                pass

        count = page.locator("a, button, [role='button']").count()
        if count == last_count:
            break
        last_count = count


def _ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for idx in range(1, 1000):
        candidate = parent / f"{stem} ({idx}){suffix}"
        if not candidate.exists():
            return candidate
    return path


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    return name


def _click_bike_mode_checkbox(page) -> bool:
    """Check the Bike Mode checkbox on the VBO export page.

    The export page HTML is:
      <input type="checkbox" name="bikeMode" id="cbBikeMode">
      <label for="cbBikeMode">Bike Mode: replace the Cornering G (Y axis) with Lean Angle</label>
    """
    for selector in ("#cbBikeMode", "input[name='bikeMode']"):
        locator = page.locator(selector)
        try:
            locator.wait_for(state="visible", timeout=5000)
            if not locator.is_checked():
                locator.check(timeout=3000)
            print("Bike Mode: enabled")
            return True
        except Exception:
            continue
    print("Bike Mode: checkbox not found – VBO will not include lean angle")
    return False


def _click_vbo_modal_button(page) -> bool:
    modal = page.locator("text=/Download Session/i").locator("..")
    if modal.count() == 0:
        modal = page.locator("[role='dialog']")
    container = modal.first if modal.count() > 0 else page

    candidates = [
        "button:has-text('VBO')",
        "a:has-text('VBO')",
        "[role='button']:has-text('VBO')",
        "img[alt*='VBO' i]",
        "img[src*='vbo' i]",
        "[data-format='vbo']",
        "[data-type='vbo']",
        "[class*='vbo' i]",
        "svg:has-text('VBO')",
        "div:has-text('VBO')",
    ]
    for selector in candidates:
        vbo_btn = container.locator(selector)
        if vbo_btn.count() == 0:
            continue
        try:
            vbo_btn.first.click(timeout=5000)
            return True
        except Exception:
            continue

    try:
        clicked = page.evaluate(
            """
() => {
  const dialog = document.querySelector('[role=dialog]') || Array.from(document.querySelectorAll('div')).find(d => (d.textContent || '').includes('Download Session'));
  if (!dialog) return false;
  const nodes = Array.from(dialog.querySelectorAll('*'));
  const target = nodes.find(n => (n.textContent || '').trim() === 'VBO');
  if (target) {
    target.click();
    return true;
  }
  return false;
}
"""
        )
        return bool(clicked)
    except Exception:
        return False


def _open_export_screen(page) -> None:
    download_icon = page.locator(
        "[title*='Download' i], [aria-label*='Download' i], img[alt*='Download' i],"
        "[class*='download' i], [class*='cloud' i]"
    )
    if download_icon.count() > 0:
        try:
            download_icon.first.click(timeout=5000)
            time.sleep(0.5)
        except Exception:
            pass
    accept = page.locator("button:has-text('I ACCEPT'), button:has-text('Accept')")
    if accept.count() > 0:
        try:
            accept.first.click(timeout=3000)
            time.sleep(0.5)
        except Exception:
            pass


def _click_download_vbo(page) -> bool:
    selectors = [
        "input[type='submit'][value='Download VBO']",
        "button:has-text('Download VBO')",
        "a:has-text('Download VBO')",
        "[role='button']:has-text('Download VBO')",
        "div:has-text('Download VBO')",
        "text=/Download VBO/i",
    ]
    for selector in selectors:
        locator = page.locator(selector)
        if locator.count() == 0:
            continue
        try:
            locator.first.scroll_into_view_if_needed(timeout=3000)
        except Exception:
            pass
        try:
            locator.first.click(timeout=5000, force=True)
            return True
        except Exception:
            continue

    try:
        clicked = page.evaluate(
            """
() => {
  const nodes = Array.from(document.querySelectorAll('button, a, [role=button], div'));
  const target = nodes.find(n => /download vbo/i.test(n.textContent || ''));
  if (target) {
    target.click();
    return true;
  }
  return false;
}
"""
        )
        return bool(clicked)
    except Exception:
        return False


def _collect_all_session_urls(page, max_sessions: int) -> list[str]:
    """Collect all session URLs from the paginated list."""
    urls = set()

    for page_num in range(1, 10):
        if len(urls) >= max_sessions:
            break

        page_url = f"{SESSIONS_URL}?page={page_num}"
        try:
            page.goto(page_url, wait_until="domcontentloaded")
        except Exception:
            break

        last_count = -1
        for _ in range(10):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(0.5)
            load_more = page.locator(
                "button:has-text('Load more'), button:has-text('Show more')"
            )
            if load_more.count() > 0 and load_more.first.is_visible():
                try:
                    load_more.first.click()
                    time.sleep(1)
                except Exception:
                    pass

            count = page.locator("a, button, [role='button']").count()
            if count == last_count:
                break
            last_count = count

        try:
            links = page.evaluate(
                """
() => {
  const urls = [];
  const links = Array.from(document.querySelectorAll('a'));
  for (const link of links) {
    const href = link.getAttribute('href') || '';
    if (href.includes('/webapp/sessions/') || href.match(/session.*\\d/i)) {
      const text = (link.textContent || '').toLowerCase();
      if (text.includes('#') || text.includes('lap') || text.includes('session')) {
        urls.push(link.href);
      }
    }
  }
  return urls;
}
"""
            )
            for link in links:
                if len(urls) < max_sessions:
                    urls.add(link)
        except Exception as exc:
            print(f"Failed to extract links from page {page_num}: {exc}")

        time.sleep(0.5)

    return list(urls)[:max_sessions]


def _debug_screenshot(page, output_dir: Path, label: str) -> None:
    """Save a debug screenshot to output_dir for troubleshooting UI issues."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / f"debug_{label}.png"
        page.screenshot(path=str(dest), full_page=False)
        print(f"  [debug] screenshot saved to {dest}")
    except Exception as exc:
        print(f"  [debug] screenshot failed: {exc}")


def download_via_clicks(
    page, output_dir: Path, overwrite: bool, max_clicks: int, tracker: DownloadTracker,
    debug: bool = False,
) -> tuple[int, int]:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0

    session_urls = _collect_all_session_urls(page, max_clicks)
    print(f"Collected {len(session_urls)} session URLs")

    for session_url in session_urls:
        session_id = extract_session_id(session_url)

        if not overwrite and tracker.is_downloaded(session_id):
            print(f"Skipped session {session_id} (already downloaded)")
            skipped += 1
            continue

        try:
            page.goto(session_url, wait_until="domcontentloaded")
        except Exception as exc:
            print(f"Failed to navigate to {session_url}: {exc}")
            continue

        try:
            # Some RaceBox UI versions expose a per-session Bike Mode toggle on
            # the session detail page before the download dialog opens.
            _open_export_screen(page)
            if not _click_vbo_modal_button(page):
                print("Failed to open VBO export screen")
                continue

            # Clicking VBO in the modal navigates to the export page; wait for
            # it to finish loading before trying to interact with the form.
            try:
                page.wait_for_load_state("domcontentloaded", timeout=10000)
            except Exception:
                time.sleep(1)

            if not _click_bike_mode_checkbox(page) and debug:
                _debug_screenshot(page, output_dir, f"bikemode_{session_id}")

            try:
                with page.expect_download(timeout=20000) as download_info:
                    if not _click_download_vbo(page):
                        raise RuntimeError("Download VBO button not found")
                download = download_info.value
                suggested = download.suggested_filename or "session.vbo"
                filename = sanitize_filename(suggested)
                if not filename.lower().endswith(".vbo"):
                    filename = filename + ".vbo" if filename else "session.vbo"
                target = output_dir / filename
                if target.exists() and not overwrite:
                    tracker.mark_downloaded(session_id, filename)
                    skipped += 1
                    print(f"Skipped {target.name} (exists)")
                else:
                    target = _ensure_unique_path(target)
                    if target.exists() and overwrite:
                        target.unlink()
                    try:
                        download.save_as(str(target))
                        if target.exists():
                            tracker.mark_downloaded(session_id, target.name)
                            downloaded += 1
                            print(f"Saved {target.name}")
                        else:
                            print(
                                f"Failed to save {target.name} (file not found after save_as)"
                            )
                    except Exception as save_exc:
                        print(f"Failed to save {target.name}: {save_exc}")
            except PlaywrightTimeoutError as exc:
                print(f"Failed to download from {session_url}: {exc}")
            except Exception as exc:
                print(f"Failed to download from {session_url}: {exc}")
        except Exception as exc:
            print(f"Error processing {session_url}: {exc}")

    return downloaded, skipped

