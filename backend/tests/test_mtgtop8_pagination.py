"""mtgtop8 event list: follow the two-week view's pagination."""

from datetime import date, timedelta

from app.jobs import mtgtop8_scrape as scrape


def day(days_ago: int) -> str:
    return (date.today() - timedelta(days=days_ago)).strftime("%d/%m/%y")


def row(event_id: str, name: str, when: str) -> str:
    return (f'<tr class=hover_tr><td></td><td class=S14><a href=event?e={event_id}&f=ST>{name}</a></td>'
            f'<td></td><td align=right class=S12>{when}</td></tr>')


def page(rows, nav: str = "") -> str:
    # an undated "Last major events" style row is on every real page
    undated = '<tr><td><a href=event?e=999&f=ST>The Decks to Beat</a></td></tr>'
    return f"<html><table>{''.join(rows)}{undated}</table><div>{nav}</div></html>"


NEXT = ('<div class=Nav_cur>1</div><div class=Nav_norm><a href=?f=ST&meta=50&cp=2>2</a></div>'
        '<div class=Nav_norm><a href=?f=ST&meta=50&cp=2>Next</a></div>')
PREV = '<div class=Nav_norm><a href=?f=ST&meta=50&cp=1>Prev</a></div>'


def fake_fetch(monkeypatch, pages):
    urls = []

    async def fetch(client, url):
        urls.append(url)
        return pages[url]

    monkeypatch.setattr(scrape, "fetch_page", fetch)
    return urls


async def test_follows_next_page_and_dedupes(monkeypatch):
    base = f"{scrape.MTGTOP8_BASE_URL}/format"
    urls = fake_fetch(monkeypatch, {
        f"{base}?f=ST": page([row("1", "League", day(1)), row("1", "League", day(1)),
                              row("2", "Challenge", day(3))], NEXT),
        f"{base}?f=ST&meta=50&cp=2": page([row("3", "RCQ", day(13))], PREV),
    })
    events = await scrape.scrape_recent_events(None, "ST", "standard", days=14)
    assert [e["mtgtop8_id"] for e in events] == ["1", "2", "3"]
    assert urls == [f"{base}?f=ST", f"{base}?f=ST&meta=50&cp=2"]
    assert events[2]["url"] == f"{scrape.MTGTOP8_BASE_URL}/event?e=3"


async def test_stops_at_a_page_without_dated_rows(monkeypatch):
    base = f"{scrape.MTGTOP8_BASE_URL}/format"
    urls = fake_fetch(monkeypatch, {
        f"{base}?f=ST": page([row("1", "League", day(1))], NEXT),
        f"{base}?f=ST&meta=50&cp=2": page([], NEXT),
    })
    events = await scrape.scrape_recent_events(None, "ST", "standard", days=14)
    assert [e["mtgtop8_id"] for e in events] == ["1"]
    assert len(urls) == 2


async def test_drops_events_older_than_the_window(monkeypatch):
    fake_fetch(monkeypatch, {
        f"{scrape.MTGTOP8_BASE_URL}/format?f=ST": page([row("1", "New", day(2)), row("2", "Old", day(20))]),
    })
    events = await scrape.scrape_recent_events(None, "ST", "standard", days=14)
    assert [e["name"] for e in events] == ["New"]


async def test_failed_later_page_keeps_earlier_events(monkeypatch):
    base = f"{scrape.MTGTOP8_BASE_URL}/format"
    pages = {f"{base}?f=ST": page([row("1", "League", day(1))], NEXT)}

    async def fetch(client, url):
        if url not in pages:
            raise RuntimeError("timed out")
        return pages[url]

    monkeypatch.setattr(scrape, "fetch_page", fetch)
    events = await scrape.scrape_recent_events(None, "ST", "standard", days=14)
    assert [e["mtgtop8_id"] for e in events] == ["1"]
