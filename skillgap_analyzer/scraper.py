from __future__ import annotations

from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class ScrapedJob:
    url: str
    text: str


def scrape_job_page(url: str, *, timeout_s: int = 20, user_agent: str | None = None) -> ScrapedJob:
    """Fetch and extract visible text from a job page.

    Notes:
    - Ensure you have permission to scrape the target site and that you respect its robots/TOS.
    - This project ships the scraper as an optional module; the Streamlit app works offline using local datasets.
    """
    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent

    resp = requests.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return ScrapedJob(url=url, text=text)

