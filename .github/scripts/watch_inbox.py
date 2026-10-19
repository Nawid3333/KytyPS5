#!/usr/bin/env python3
"""Fork-only: report new upstream activity that concerns WATCH_USER in the fork's "Watch inbox" issue.

Run by .github/workflows/watch-inbox.yml. Standard library only. Each run reads the
checkpoint (the time window start and the PR conditions already reported) from hidden
markers in the inbox issue body, collects upstream comments, reviews and PR state changes
since then, posts at most one comment and moves the checkpoint to the run start.

Links use www.github.com (redirects to github.com): GitHub does not turn those into
cross-references, so the upstream threads get no "mentioned this" timeline entries.
"""

import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"
UPSTREAM = os.environ.get("UPSTREAM_REPO", "KytyPS5/KytyPS5")
FORK = os.environ.get("GITHUB_REPOSITORY", "Nawid3333/KytyPS5")
USER = os.environ.get("WATCH_USER", "Nawid3333")
WATCH_FILE = os.environ.get("WATCH_FILE", ".github/watch-threads.txt")
DRY_RUN = os.environ.get("DRY_RUN") == "1"

INBOX_TITLE = "Watch inbox"
INBOX_LABEL = "watch-inbox"
FIRST_RUN_WINDOW = dt.timedelta(hours=6)
MAX_WINDOW = dt.timedelta(days=3)
# Events newer than "now - SETTLE" wait for the next run, so an event that is not yet
# listed by the API (clock skew, eventual consistency) is never skipped.
SETTLE = dt.timedelta(seconds=60)
MAX_SHOWN = 15
MAX_LISTED = 200
MERGE_CHECKS_PER_RUN = 8
MIN_RATE_REMAINING = 100
MAX_PAGES = 10
FAILED = {"failure", "timed_out", "startup_failure"}

SINCE_RE = re.compile(r"<!-- watch-since: (\S+) -->")
STATE_RE = re.compile(r"<!-- state: (#\d+ [a-z-]+ [0-9a-f]+) -->")
MENTION_RE = re.compile(r"@" + re.escape(USER) + r"\b", re.IGNORECASE)


class RateLow(Exception):
    pass


class Api:
    def __init__(self, token):
        self.token = token
        self.calls = 0
        self.remaining = None

    def request(self, method, url, params=None, body=None):
        if not url.startswith("https://"):
            url = API + url
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        if self.remaining is not None and self.remaining < MIN_RATE_REMAINING:
            raise RateLow(f"rate limit remaining {self.remaining} < {MIN_RATE_REMAINING}")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "watch-inbox")
        req.add_header("Authorization", "Bearer " + self.token)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        for attempt in (1, 2):  # one retry for transient server and network errors
            self.calls += 1
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    headers = resp.headers
                    raw = resp.read()
                break
            except urllib.error.HTTPError as err:
                self._note_rate(err.headers)
                if err.code >= 500 and attempt == 1:
                    time.sleep(5)
                    continue
                detail = err.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"{method} {url} -> HTTP {err.code}: {detail}") from None
            except urllib.error.URLError:
                if attempt == 2:
                    raise
                time.sleep(5)
        self._note_rate(headers)
        return (json.loads(raw) if raw else None), headers

    def _note_rate(self, headers):
        value = headers.get("X-RateLimit-Remaining") if headers else None
        if value is not None:
            self.remaining = int(value)

    def get(self, path, params=None):
        return self.request("GET", path, params)[0]

    def pages(self, path, params):
        url, query, page = path, dict(params, per_page=100), 0
        while url and page < MAX_PAGES:
            data, headers = self.request("GET", url, query)
            yield from data
            url, query, page = next_link(headers, "next"), None, page + 1

    def tail(self, path, pages=2):
        """Items of the last `pages` pages of an ascending list, oldest first."""
        data, headers = self.request("GET", path, {"per_page": 100})
        last = next_link(headers, "last")
        if not last:
            return data
        data, headers = self.request("GET", last)
        if pages > 1 and next_link(headers, "prev"):
            data = self.request("GET", next_link(headers, "prev"))[0] + data
        return data


def next_link(headers, rel):
    for part in (headers.get("Link") or "").split(","):
        match = re.match(r'\s*<([^>]+)>;\s*rel="([^"]+)"', part)
        if match and match.group(2) == rel:
            return match.group(1)
    return None


def parse_ts(text):
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def iso(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


GHOST = {"login": "ghost", "type": "User"}


def same_user(login):
    return (login or "").lower() == USER.lower()


def is_bot(user):
    return user.get("type") == "Bot" or user.get("login", "").endswith("[bot]")


def www(url):
    return url.replace("https://github.com/", "https://www.github.com/", 1)


def load_watch_list(path):
    numbers = set()
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                numbers.update(int(n) for n in re.findall(r"\d+", line.split("#", 1)[0]))
    except FileNotFoundError:
        print(f"::warning::{path} not found; watching only own threads and mentions")
    return numbers


# ---- text handling --------------------------------------------------------------------


def strip_markup(body):
    text = re.sub(r"<!--.*?-->", " ", body or "", flags=re.S)
    previous = None
    while previous != text:  # innermost <details> first, CodeRabbit nests them
        previous = text
        text = re.sub(r"<details>(?:(?!<details>).)*?</details>", " ", text, flags=re.S | re.I)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # images and badges
    text = re.sub(r"\[\s*\]\([^)]*\)", " ", text)  # links left empty by that
    text = re.sub(r"```.*?(```|$)", " ", text, flags=re.S)  # code blocks
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def plain(line):
    line = re.sub(r"^\s*#+\s*", "", line)
    line = line.replace("**", "").replace("__", "")
    line = re.sub(r"(?<!\w)_(\S(?:.*?\S)?)_(?!\w)", r"\1", line)
    return line.strip(" -*|")


def summary(body, bot):
    lines = []
    for line in strip_markup(body).splitlines():
        line = line.strip()
        if line.startswith(">"):
            if not bot:
                continue  # a human quoting earlier text
            line = re.sub(r"^(>\s*)+", "", line)  # bots put their headline in a quote
        line = re.sub(r"^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]", "", line).strip()
        if line and not re.fullmatch(r"[-*_=|: ]+", line):
            lines.append(line)
    if not lines:
        return ""
    if not bot:
        return " ".join(lines)
    # CodeRabbit: inline comments open with a tag line ("Potential issue | Major") and a
    # bold title; reviews and summaries carry a bold headline in their first lines.
    if "|" in lines[0] and len(lines) > 1:
        return f"{plain(lines[0])} · {plain(lines[1])}"
    for line in lines[:4]:
        if re.fullmatch(r"\*\*[^*]+\*\*", line):
            return plain(line)
    first = plain(lines[0])
    if len(first) < 40 and len(lines) > 1:
        first = f"{first} · {plain(lines[1])}"
    return first


def safe(text, limit):
    """One markdown line that mentions, links and references nobody."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # markdown links: keep the text
    text = re.sub(r"https?://\S+", "(link)", text)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    text = text.replace("<", "&lt;").replace(">", "&gt;").replace("[", "\\[").replace("]", "\\]")
    text = text.replace("@", "@\u200b")
    text = re.sub(r"#(?=\d)", "#\u200b", text)
    return text


# ---- inbox and checkpoint -------------------------------------------------------------


def find_inbox(api):
    def pick(issues):
        found = [i for i in issues if i["title"] == INBOX_TITLE and "pull_request" not in i]
        found.sort(key=lambda i: (i["state"] != "open", i["number"]))
        return found[0] if found else None

    inbox = pick(api.pages(f"/repos/{FORK}/issues", {"labels": INBOX_LABEL, "state": "all"}))
    if inbox is None:
        inbox = pick(api.pages(f"/repos/{FORK}/issues", {"state": "all"}))
    if inbox is None and not DRY_RUN:
        try:
            api.request("POST", f"/repos/{FORK}/labels",
                        body={"name": INBOX_LABEL, "color": "0e8a16",
                              "description": "Fork-only upstream watch inbox"})
        except RuntimeError as err:
            if "HTTP 422" not in str(err):
                raise
        inbox = api.request("POST", f"/repos/{FORK}/issues", body={
            "title": INBOX_TITLE, "body": inbox_body("", [], 0), "labels": [INBOX_LABEL]})[0]
        print(f"created inbox issue #{inbox['number']}")
    return inbox


def read_checkpoint(api, inbox, now):
    body = (inbox or {}).get("body") or ""
    since = SINCE_RE.search(body)
    if since is None and inbox and inbox.get("comments"):
        for comment in reversed(api.tail(f"/repos/{FORK}/issues/{inbox['number']}/comments")):
            if SINCE_RE.search(comment["body"] or ""):
                body = comment["body"]
                since = SINCE_RE.search(body)
                break
    if since is None:
        return now - FIRST_RUN_WINDOW, set()
    return max(parse_ts(since.group(1)), now - MAX_WINDOW), set(STATE_RE.findall(body))


def markers(since, state):
    lines = [f"<!-- state: {key} -->" for key in sorted(state)]
    return "\n".join(lines + [f"<!-- watch-since: {since} -->"])


def inbox_body(since, state, calls):
    text = (
        "Fork-only. `.github/workflows/watch-inbox.yml` posts one comment here whenever "
        f"{UPSTREAM} has new comments, reviews or PR state changes on threads listed in "
        f"`.github/watch-threads.txt`, on {USER}'s open issues and PRs, or that mention {USER}."
    )
    if since:
        text += f"\n\nLast check: window up to {since.replace('T', ' ').rstrip('Z')} UTC, {calls} API calls."
        text += "\n\n" + markers(since, state)
    return text


# ---- collection -----------------------------------------------------------------------


class Collector:
    def __init__(self, api, since, until, watch):
        self.api, self.since, self.until, self.watch = api, since, until, watch
        self.threads = {}
        self.events = []

    def in_window(self, stamp):
        return stamp is not None and self.since <= parse_ts(stamp) < self.until

    def remember(self, item):
        pr = item.get("pull_request") or {}
        self.threads[item["number"]] = {
            "title": item["title"],
            "author": (item.get("user") or GHOST)["login"],
            "open": item["state"] == "open",
            "closed_at": item.get("closed_at"),
            "merged": bool(pr.get("merged_at")) or bool(item.get("merged_at")),
            "url": item["html_url"],
        }

    def thread(self, number):
        if number not in self.threads:
            self.remember(self.api.get(f"/repos/{UPSTREAM}/issues/{number}"))
        return self.threads[number]

    def wanted(self, number, body):
        if number in self.watch or MENTION_RE.search(body or ""):
            return True
        info = self.thread(number)
        return info["open"] and same_user(info["author"])

    def add(self, number, stamp, kind, url, user=None, body=None):
        bot = bool(user) and is_bot(user)
        author = user["login"].removesuffix("[bot]") if user else ""
        self.events.append({"n": number, "at": stamp, "kind": kind, "url": url,
                            "author": author, "bot": bot,
                            "text": summary(body, bot) if body is not None else ""})

    def threads_since(self):
        params = {"state": "all", "sort": "updated", "direction": "desc", "since": iso(self.since)}
        for item in self.api.pages(f"/repos/{UPSTREAM}/issues", params):
            self.remember(item)

    def comments(self):
        params = {"since": iso(self.since), "sort": "created", "direction": "asc"}
        for c in self.api.pages(f"/repos/{UPSTREAM}/issues/comments", params):
            c["user"] = c.get("user") or GHOST
            number = int(c["issue_url"].rsplit("/", 1)[1])
            if self.in_window(c["created_at"]) and not same_user(c["user"]["login"]) \
                    and self.wanted(number, c["body"]):
                self.add(number, c["created_at"], "comment", c["html_url"], c["user"], c["body"])

    def review_comments(self):
        params = {"since": iso(self.since), "sort": "created", "direction": "asc"}
        for c in self.api.pages(f"/repos/{UPSTREAM}/pulls/comments", params):
            c["user"] = c.get("user") or GHOST
            number = int(c["pull_request_url"].rsplit("/", 1)[1])
            if self.in_window(c["created_at"]) and not same_user(c["user"]["login"]) \
                    and self.wanted(number, c["body"]):
                where = c["path"].rsplit("/", 1)[-1]
                line = c.get("line") or c.get("original_line")
                kind = f"inline {where}:{line}" if line else f"inline {where}"
                self.add(number, c["created_at"], kind, c["html_url"], c["user"], c["body"])

    def closures(self):
        for number, info in self.threads.items():
            if self.in_window(info["closed_at"]) and not info["open"] \
                    and (number in self.watch or same_user(info["author"])):
                self.add(number, info["closed_at"], "merged" if info["merged"] else "closed",
                         info["url"])

    def own_prs(self, previous):
        """Reviews on USER's open PRs, plus conflicts and failed checks not reported yet."""
        prs = [p for p in self.api.pages(f"/repos/{UPSTREAM}/pulls", {"state": "open"})
               if same_user(p["user"]["login"])]
        prs.sort(key=lambda p: p["number"])
        runs = self.api.get(f"/repos/{UPSTREAM}/actions/runs",
                            {"actor": USER, "per_page": 100})["workflow_runs"]
        recent = self.since - dt.timedelta(hours=1)
        rounds = max(1, -(-len(prs) // MERGE_CHECKS_PER_RUN))
        turn = int(self.until.timestamp() // 600) % rounds
        state = set()
        for index, pr in enumerate(prs):
            number, sha = pr["number"], pr["head"]["sha"][:7]
            self.threads.setdefault(number, {
                "title": pr["title"], "author": pr["user"]["login"], "open": True,
                "closed_at": None, "merged": False, "url": pr["html_url"]})
            updated = parse_ts(pr["updated_at"])
            if updated >= self.since:
                self.reviews(number)
            failed = self.failed_runs(runs, pr["head"]["sha"])
            if updated >= recent:
                status = self.api.get(f"/repos/{UPSTREAM}/commits/{pr['head']['sha']}/status")
                if status["state"] in ("failure", "error"):
                    failed += [s["context"] for s in status["statuses"]
                               if s["state"] in ("failure", "error")]
            elif f"#{number} checks-failed {sha}" in previous:
                failed = failed or ["status"]
            if failed:
                key = f"#{number} checks-failed {sha}"
                state.add(key)
                if key not in previous:
                    names = ", ".join(sorted(set(failed)))
                    self.add(number, iso(self.until), f"checks failed ({names}), head {sha}",
                             pr["html_url"] + "/checks")
            key = f"#{number} dirty {sha}"
            if updated >= recent or index % rounds == turn:
                merge_state = self.api.get(f"/repos/{UPSTREAM}/pulls/{number}").get("mergeable_state")
                dirty = merge_state == "dirty" or (merge_state in (None, "unknown") and key in previous)
            else:
                dirty = key in previous
            if dirty:
                state.add(key)
                if key not in previous:
                    self.add(number, iso(self.until), f"merge conflict with main, head {sha}",
                             pr["html_url"])
        return state

    def failed_runs(self, runs, sha):
        latest = {}
        for run in runs:
            if run["head_sha"] == sha and run["status"] == "completed":
                if run["workflow_id"] not in latest or run["id"] > latest[run["workflow_id"]]["id"]:
                    latest[run["workflow_id"]] = run
        return [run["name"] for run in latest.values() if run["conclusion"] in FAILED]

    def reviews(self, number):
        for review in self.api.tail(f"/repos/{UPSTREAM}/pulls/{number}/reviews"):
            body = review.get("body") or ""
            review["user"] = review.get("user") or GHOST
            if not self.in_window(review.get("submitted_at")) or same_user(review["user"]["login"]):
                continue
            if review["state"] == "COMMENTED" and not body.strip():
                continue  # container of inline comments, those are reported one by one
            kind = {"APPROVED": "review approved", "CHANGES_REQUESTED": "review changes requested",
                    "DISMISSED": "review dismissed"}.get(review["state"], "review")
            self.add(number, review["submitted_at"], kind, review["html_url"], review["user"], body)


# ---- output ---------------------------------------------------------------------------


def render(events, threads, since, until, state):
    order = {}
    for event in sorted(events, key=lambda e: e["at"]):
        first = order.setdefault(event["n"], [1, event["at"]])
        if not event["bot"]:
            first[0] = 0  # threads with human comments or state changes come first
    events = sorted(events, key=lambda e: (order[e["n"]], e["n"], e["at"]))
    lines = []
    for event in events[:MAX_LISTED]:
        title = safe(threads[event["n"]]["title"], 50)
        head = f"- #\u200b{event['n']} {title} —"
        if event["author"]:
            text = safe(event["text"], 160)
            lines.append(f"{head} {event['author']} {event['kind']}: {text} [link]({www(event['url'])})")
        else:
            lines.append(f"{head} {event['kind']} [link]({www(event['url'])})")
    span = f"{since.strftime('%m-%d %H:%M')}–{until.strftime('%H:%M')} UTC"
    out = [f"**{len(events)} new** · {span}", ""] + lines[:MAX_SHOWN]
    if len(lines) > MAX_SHOWN:
        out += ["", f"<details><summary>{len(events) - MAX_SHOWN} more</summary>", ""]
        out += lines[MAX_SHOWN:]
        if len(events) > MAX_LISTED:
            out.append(f"- … {len(events) - MAX_LISTED} not listed")
        out += ["", "</details>"]
    return "\n".join(out + ["", markers(iso(until), state)])


def main():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN is not set")
    api = Api(token)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    until = now - SETTLE
    try:
        inbox = find_inbox(api)
        since, previous = read_checkpoint(api, inbox, until)
        since = min(since, until)
        print(f"inbox #{inbox['number'] if inbox else '-'}, window {iso(since)} .. {iso(until)}, "
              f"{len(previous)} known conditions")
        collect = Collector(api, since, until, load_watch_list(WATCH_FILE))
        collect.threads_since()
        collect.comments()
        collect.review_comments()
        collect.closures()
        state = collect.own_prs(previous)
    except RateLow as err:
        print(f"::warning::stopped early, nothing posted, window kept: {err}")
        return
    comment = render(collect.events, collect.threads, since, until, state) if collect.events else None
    calls = api.calls + (1 if comment else 0) + 1
    body = inbox_body(iso(until), state, calls)
    if DRY_RUN or inbox is None:
        print(comment or "(nothing new)")
        print("---- inbox body ----\n" + body)
    else:
        if comment:  # post before moving the checkpoint: a failed run repeats, never loses
            api.request("POST", f"/repos/{FORK}/issues/{inbox['number']}/comments", body={"body": comment})
        api.request("PATCH", f"/repos/{FORK}/issues/{inbox['number']}", body={"body": body})
    summary_line = (f"{len(collect.events)} events, {'posted' if comment else 'nothing posted'}, "
                    f"{api.calls} API calls, rate limit remaining {api.remaining}")
    print(summary_line)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(f"Watch inbox: {summary_line}\n")


if __name__ == "__main__":
    main()
