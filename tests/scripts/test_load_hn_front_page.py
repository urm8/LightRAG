from scripts.load_hn_front_page import (
  extract_hn_external_links,
  filter_external_source_urls,
  parse_front_page,
  should_use_discussion_page,
)


def test_extract_hn_external_links_includes_comment_links() -> None:
    html = """
    <html>
      <body>
        <span class="titleline">
          <a href="https://ianthehenry.com/posts/why-janet/">Why Janet?</a>
          <a href="from?site=ianthehenry.com">ianthehenry.com</a>
        </span>
        <table>
          <tr class="athing comtr">
            <td class="default">
              <div class="comment">
                <span class="commtext c00">
                  <p>Source mirror: <a href="https://janet-lang.org/docs/bindings.html">bindings</a></p>
                  <p>Context: <a href="https://janet.guide/">guide</a></p>
                  <p>Internal: <a href="item?id=35539255">HN thread</a></p>
                </span>
              </div>
            </td>
          </tr>
        </table>
      </body>
    </html>
    """

    assert extract_hn_external_links(html) == [
      "https://ianthehenry.com/posts/why-janet/",
      "https://janet-lang.org/docs/bindings.html",
      "https://janet.guide/",
    ]


def test_filter_external_source_urls_excludes_primary_and_discussion_urls() -> None:
    html = """
    <html>
      <body>
        <span class="titleline">
          <a href="https://www.economist.com/story">Economist story</a>
        </span>
        <div class="toptext">
          <a href="https://archive.ph/nKEVw">mirror</a>
        </div>
        <span class="commtext c00">
          <a href="https://www.wsj.com/finance/story">secondary source</a>
          <a href="https://news.ycombinator.com/item?id=123">HN internal</a>
        </span>
      </body>
    </html>
    """

    filtered = filter_external_source_urls(
        extract_hn_external_links(html),
        exclude_urls=[
            "https://www.economist.com/story",
            "https://news.ycombinator.com/item?id=48364055",
        ],
    )

    assert filtered == [
        "https://archive.ph/nKEVw",
        "https://www.wsj.com/finance/story",
    ]


def test_parse_front_page_uses_board_title_link_as_primary_url() -> None:
    html = """
    <html>
      <body>
        <table>
          <tr class="athing" id="123">
            <td class="title">
              <span class="titleline">
                <a href="https://example.com/story">Example story</a>
                <span class="sitebit comhead">(<a href="from?site=example.com">example.com</a>)</span>
              </span>
            </td>
          </tr>
          <tr>
            <td class="subtext"><a href="item?id=123">42 comments</a></td>
          </tr>
          <tr class="athing" id="456">
            <td class="title">
              <span class="titleline">
                <a href="item?id=456">Ask HN: Example</a>
              </span>
            </td>
          </tr>
          <tr>
            <td class="subtext"><a href="item?id=456">12 comments</a></td>
          </tr>
        </table>
      </body>
    </html>
    """

    posts = parse_front_page(html, 10)

    assert len(posts) == 2
    assert posts[0].article_url == "https://example.com/story"
    assert posts[0].discussion_url == "https://news.ycombinator.com/item?id=123"
    assert not should_use_discussion_page(posts[0])

    assert posts[1].article_url == "https://news.ycombinator.com/item?id=456"
    assert should_use_discussion_page(posts[1])