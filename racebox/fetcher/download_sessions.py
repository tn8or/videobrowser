#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

LOGIN_URL = "https://www.racebox.pro/webapp/login"
SESSIONS_URL = "https://www.racebox.pro/webapp/sessions"


def read_secret(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Empty secret in {path}")
    return value


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


def read_secret(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Empty secret in {path}")
    return value


def find_first_visible(page, selectors: Iterable[str]):
    for selector in selectors:
        locator = page.locator(selector)
        if locator.count() > 0 and locator.first.is_visible():
            return locator.first
    return None


def login(page, username: str, password: str) -> None:
    page.goto(LOGIN_URL, wait_until="domcontentloaded")

    accept = page.locator("button:has-text('I ACCEPT'), button:has-text('Accept')")
    if accept.count() > 0:
        try:
            accept.first.click(timeout=3000)
            time.sleep(0.5)
        except Exception:
            pass

    user_input = find_first_visible(
        page,
        [
            "input[type='email']",
            "input[name='email']",
            "input[autocomplete='username']",
            "input[name='username']",
        ],
    )
    if not user_input:
        raise RuntimeError("Could not find username/email input on login page")
    user_input.fill(username)

    pass_input = find_first_visible(
        page,
        [
            "input[type='password']",
            "input[name='password']",
            "input[autocomplete='current-password']",
        ],
    )
    if not pass_input:
        raise RuntimeError("Could not find password input on login page")
    pass_input.fill(password)

    submit = find_first_visible(
        page,
        [
            "button:has-text('Log in')",
            "button:has-text('Login')",
            "button:has-text('Sign in')",
            "input[type='submit']",
        ],
    )
    if submit:
        submit.click()
    else:
        pass_input.press("Enter")

    try:
        page.wait_for_url("**/sessions**", timeout=20000)
    except PlaywrightTimeoutError:
        page.goto(SESSIONS_URL, wait_until="domcontentloaded")


def try_enable_bike_mode(page) -> None:
    candidates = [
        "text=/Bike Mode/i",
        "text=/Bike/i",
        "[role='button']:has-text('Bike')",
        "button:has-text('Bike')",
    ]
    for selector in candidates:
        locator = page.locator(selector)
        if locator.count() == 0:
            continue
        try:
            if locator.first.is_visible():
                locator.first.click()
                time.sleep(1)
                return
        except Exception:
            continue


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


def collect_session_items(page) -> list:
    links = page.evaluate(
        """
() => {
    const sessions = [];
    const items = Array.from(document.querySelectorAll("[role='listitem'], tr, .session-item, [class*='session'], [class*='list-item']"));
    for (const item of items) {
        const link = item.querySelector("a");
        if (link && link.href) {
            sessions.push({url: link.href, text: item.textContent});
        } else if (item.onclick || (item.classList && Array.from(item.classList).some(c => c.includes('clickable') || c.includes('item')))) {
            sessions.push({element: item, text: item.textContent});
        }
    }
    return sessions;
}
"""
    )
    result = []
    for item in links:
        if "url" in item:
            result.append(urljoin(page.url, item["url"]))
    return result


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


def filename_from_headers(headers: dict, url: str) -> str:
    content_disp = headers.get("content-disposition") or headers.get(
        "Content-Disposition"
    )
    if content_disp:
        match = re.search(
            r"filename\*=UTF-8''([^;]+)|filename=\"?([^;\"]+)\"?", content_disp
        )
        if match:
            name = match.group(1) or match.group(2)
            if name:
                return name
    parsed = urlparse(url)
    name = Path(parsed.path).name or "session.vbo"
    return name


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    return name


def download_all(
    context, urls: list[str], output_dir: Path, overwrite: bool
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0

    for url in urls:
        try:
            response = context.request.get(url)
        except Exception as exc:
            print(f"Failed to fetch {url}: {exc}")
            continue

        if not response.ok:
            print(f"Non-200 for {url}: {response.status}")
            continue

        filename = sanitize_filename(filename_from_headers(response.headers, url))
        target = output_dir / filename
        if target.exists() and not overwrite:
            skipped += 1
            continue

        try:
            target.write_bytes(response.body())
            downloaded += 1
            print(f"Saved {target.name}")
        except Exception as exc:
            print(f"Failed to write {target}: {exc}")

    return downloaded, skipped


def _session_rows_locator(page):
    return (
        page.locator("div, tr")
        .filter(has_text=re.compile(r"#\d+", re.IGNORECASE))
        .filter(has_text=re.compile(r"\d{1,2}:\d{2}|laps?", re.IGNORECASE))
    )


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


def _go_next_page(page) -> bool:
    next_btn = page.locator("text='»', text=/Next/i, button:has-text('>')")
    if next_btn.count() == 0:
        return False
    try:
        if next_btn.first.is_visible():
            next_btn.first.click()
            time.sleep(1)
            return True
    except Exception:
        return False
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download RaceBox VBO sessions (Bike Mode)"
    )
    parser.add_argument(
        "--output-dir", default=".", help="Directory to save .vbo files"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--headful", action="store_true", help="Run browser in headful mode"
    )
    parser.add_argument(
        "--max-clicks",
        type=int,
        default=200,
        help="Max download/export buttons to click",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug screenshots when UI elements are not found",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    username = read_secret(script_dir / "username")
    password = read_secret(script_dir / "password")
    output_dir = Path(args.output_dir).resolve()
    tracker = DownloadTracker(script_dir / "downloads.db")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        login(page, username, password)
        load_all_sessions(page)

        print("Clicking through sessions to download VBO files...")
        downloaded, skipped = download_via_clicks(
            page, output_dir, args.overwrite, args.max_clicks, tracker,
            debug=args.debug,
        )
        print(f"Downloaded: {downloaded}, Skipped: {skipped}")

        browser.close()

    return 0 if downloaded > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
