"""Crawl scope. Edit these lists rather than the crawler logic."""

# Pages to start from. The crawler follows links from these.
SEEDS = [
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/",
    "https://datascience.uchicago.edu/education/masters-programs/in-person-program/",
    "https://datascience.uchicago.edu/education/masters-programs/online-program/",
    "https://datascience.uchicago.edu/education/tuition-fees-aid/",
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faqs/",
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/how-to-apply/",
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/events-deadlines/",
]

# A link is followed (crawled AND its links explored) if its URL starts with one of these.
# Note: tuition and the in-person/online pages live OUTSIDE the MS-ADS path,
# so scoping to the MS-ADS path alone would miss the tuition answer.
IN_SCOPE_PREFIXES = [
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/",
    "https://datascience.uchicago.edu/education/masters-programs/in-person-program/",
    "https://datascience.uchicago.edu/education/masters-programs/online-program/",
    "https://datascience.uchicago.edu/education/tuition-fees-aid/",
]

# Leaf pages: fetched if linked from an in-scope page, but their links are NOT followed.
# These hold answers the eval set asks about (English scores, fee waiver, MBA/MS).
LEAF_PREFIXES = [
    "https://grad.uchicago.edu/admissions/apply/english-language-requirements/",
    "https://physicalsciences.uchicago.edu/academics/",  # PSD application requirements / fee waiver
    # (was the whole PSD site, which pulled in irrelevant pages such as alumni-friends)
    "https://www.chicagobooth.edu/programs/joint-degree-programs/",  # MBA/MS
    "https://ms-ads.datascience.uchicago.edu/",
]

# Never crawl URLs containing these (event pages, news, feeds, WP internals).
EXCLUDE_PATTERNS = [
    "/events/", "/news/", "/feed", "/wp-json", "/wp-admin", "/wp-content/",
    "/tag/", "/category/", "/author/", "?share=", "/page/",
]
EXCLUDE_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip",
                      ".mp4", ".mp3", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx")

MAX_DEPTH = 3
MAX_PAGES = 200
REQUEST_DELAY_SEC = 1.0
TIMEOUT_SEC = 20
USER_AGENT = "Mozilla/5.0 (compatible; MSADS-RAG-ClassProject/1.0)"

# If the visible text of a fetched page is shorter than this, it is probably JS-rendered.
MIN_TEXT_CHARS = 300
