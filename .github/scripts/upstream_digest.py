#!/usr/bin/env python3
"""Fork-only: keep a rolling 48-hour list of ALL upstream activity in the fork's "Upstream digest" issue body.

Run by .github/workflows/upstream-digest.yml every 10 minutes. Standard library only; the
GitHub client and the text helpers come from watch_inbox.py. watch_inbox.py reports only what
concerns WATCH_USER, as comments; this digest lists everything and never comments: each run
rewrites the issue body, so it sends no notifications. A reader needs one call (the issue body)
to see everything; keyed by the item lines, it can print only the lines it has not seen.

Each run reads its state from the body: the end of the last window and the main head it saw
(hidden markers) and the listed items (one line each, "- MM-DD HH:MM ..."). It collects what
happened upstream since then - commits that reached main (compare against the remembered head,
so a commit pushed long after it was made is still found), opened/merged/closed pull requests
and issues, comments, inline comments and reviews in any thread, new discussions and their
comments (GraphQL) - drops items older than 48 hours and, when the body would pass GitHub's
65,536-character limit (counted in UTF-8 bytes, to be safe), the oldest items. Items by
WATCH_USER are tagged *own*, bot items *bot*.

A normal run uses about 7-12 API calls (one per list, one per pull request updated in the
window for its reviews, capped); the first fill of an empty issue about 30.

Links use www.github.com (redirects to github.com): GitHub does not turn those into
cross-references, so the upstream threads get no "mentioned this" timeline entries.

    DRY_RUN=1 GITHUB_TOKEN=... python3 -I .github/scripts/upstream_digest.py   # print the body, write nothing
"""

import datetime as dt
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from watch_inbox import GHOST, Api, RateLow, is_bot, iso, parse_ts, safe, summary, www  # noqa: E402

UPSTREAM = os.environ.get("UPSTREAM_REPO", "KytyPS5/KytyPS5")
FORK = os.environ.get("GITHUB_REPOSITORY", "Nawid3333/KytyPS5")
USER = os.environ.get("WATCH_USER", "Nawid3333")
DRY_RUN = os.environ.get("DRY_RUN") == "1"

DIGEST_TITLE = "Upstream digest"
DIGEST_LABEL = "upstream-digest"
WINDOW = dt.timedelta(hours=48)
SETTLE = dt.timedelta(seconds=60)
BODY_LIMIT = 65536
BODY_BUDGET = BODY_LIMIT - 1536
REVIEW_PRS_PER_RUN = 20
TITLE_LOOKUPS_PER_RUN = 10
TITLE_LEN = 40
EXCERPT_LEN = 120
BOT_LOGINS = {"coderabbitai", "github-actions", "dependabot", "copilot", "renovate"}

SECTIONS = (
    ("commits", "Commits on main"),
    ("prs", "Pull requests"),
    ("issues", "Issues"),
    ("comments", "Comments and reviews"),
    ("discussions", "Discussions"),
)
SECTION_OF = {title: key for key, title in SECTIONS}

SINCE_RE = re.compile(r"<!-- digest-since: (\S+) -->")
HEAD_RE = re.compile(r"<!-- main-head: ([0-9a-f]{40}) -->")
DROPPED_RE = re.compile(r"<!-- dropped: (\d+) (\S+) -->")
HEADER_RE = re.compile(r"^### (.+?)(?: \(\d+\))?$")
LINE_RE = re.compile(r"^- (\d\d)-(\d\d) (\d\d):(\d\d) (?:\*(own|bot)\* )?(.+)$")
URL_RE = re.compile(r"\]\((https://[^)\s]+)\)")

DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    hasDiscussionsEnabled
    discussions(first: 15, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number title url createdAt updatedAt
        author { login __typename }
        category { name }
        comments(last: 30) {
          nodes {
            url createdAt body
            author { login __typename }
            replies(last: 20) { nodes { url createdAt body author { login __typename } } }
          }
        }
      }
    }
  }
}
"""

REVIEW_KINDS = {"APPROVED": "approved", "CHANGES_REQUESTED": "requested changes",
                "DISMISSED": "review dismissed"}


# ---- items ----------------------------------------------------------------------------


def item(section, at, tag, text):
    """One digest entry: its section, time (datetime, UTC), tag ("own", "bot" or "") and markdown text."""
    if isinstance(at, str):
        at = parse_ts(at)
    return {"section": section, "at": at, "tag": tag, "text": text}


def line(entry):
    tag = f"*{entry['tag']}* " if entry["tag"] else ""
    return f"- {entry['at'].strftime('%m-%d %H:%M')} {tag}{entry['text']}"


def key(entry):
    """Identity for de-duplication: the section, the first link and the first word."""
    url = URL_RE.search(entry["text"])
    return entry["section"], url.group(1) if url else "", entry["text"].split(" ", 1)[0]


def minute(moment):
    return moment.replace(second=0, microsecond=0)


def name(user):
    return safe((user or GHOST)["login"].removesuffix("[bot]"), 39)


def tag_of(user):
    user = user or GHOST
    login = user.get("login", "")
    if login.lower() == USER.lower():
        return "own"
    if is_bot(user) or user.get("__typename") == "Bot" or login.removesuffix("[bot]").lower() in BOT_LOGINS:
        return "bot"
    return ""


def thread_ref(number, url, title):
    return f"[#{number}]({www(url)}) {safe(title, TITLE_LEN)}"


def in_window(stamp, since, until):
    return bool(stamp) and since <= parse_ts(stamp) < until


def thread_events(issue, since, until):
    """Opened, merged and closed events of one /issues list entry inside the window."""
    pr = issue.get("pull_request")
    section = "prs" if pr is not None else "issues"
    user = issue.get("user") or GHOST
    ref = thread_ref(issue["number"], issue["html_url"], issue["title"])
    events = []
    if in_window(issue.get("created_at"), since, until):
        events.append(item(section, issue["created_at"], tag_of(user), f"opened {ref} — by {name(user)}"))
    if issue.get("state") == "closed" and in_window(issue.get("closed_at"), since, until):
        if pr is not None:
            verb = "merged" if pr.get("merged_at") else "closed unmerged"
        else:
            verb = "closed as not planned" if issue.get("state_reason") == "not_planned" else "closed"
        events.append(item(section, issue["closed_at"], "", f"{verb} {ref} — from {name(user)}"))
    return events


def comment_item(comment, kind, number, title, stamp=None):
    user = comment.get("user") or comment.get("author") or GHOST
    tag = tag_of(user)
    text = safe(summary(comment.get("body"), tag == "bot"), EXCERPT_LEN)
    head = f"{name(user)} {kind} {thread_ref(number, comment.get('html_url') or comment['url'], title)}"
    return item("comments", stamp or comment["created_at"], tag, f"{head}: {text}" if text else head)


def commit_item(commit, since, until):
    """A commit that reached main in the window; its time is the commit time, kept inside the window."""
    made = parse_ts(commit["commit"]["committer"]["date"])
    at = min(max(made, since), until - dt.timedelta(seconds=1))
    user = commit.get("author")
    who = name(user) if user else safe(commit["commit"]["author"]["name"], 39)
    subject = commit["commit"]["message"].split("\n", 1)[0]
    text = f"[{commit['sha'][:7]}]({www(commit['html_url'])}) {who}: {safe(subject, 100)}"
    return item("commits", at, tag_of(user) if user else "", text)


def discussion_items(nodes, since, until):
    """New discussions, comments and replies from the GraphQL discussions nodes."""
    out = []
    for disc in nodes or []:
        if disc.get("updatedAt") and parse_ts(disc["updatedAt"]) < since:
            continue
        title, number = disc["title"], disc["number"]
        if in_window(disc.get("createdAt"), since, until):
            author = disc.get("author") or GHOST
            category = f" ({safe(disc['category']['name'], 30)})" if disc.get("category") else ""
            out.append(item("discussions", disc["createdAt"], tag_of(author),
                            f"opened {thread_ref(number, disc['url'], title)} — by {name(author)}{category}"))
        for comment in (disc.get("comments") or {}).get("nodes") or []:
            entries = [("comment", comment)]
            entries += [("reply", reply) for reply in (comment.get("replies") or {}).get("nodes") or []]
            for kind, entry in entries:
                if in_window(entry.get("createdAt"), since, until):
                    author = entry.get("author") or GHOST
                    tag = tag_of(author)
                    text = safe(summary(entry.get("body"), tag == "bot"), EXCERPT_LEN)
                    head = f"{name(author)} {kind} {thread_ref(number, entry['url'], title)}"
                    out.append(item("discussions", entry["createdAt"], tag, f"{head}: {text}" if text else head))
    return out


# ---- body: parse, merge, render -------------------------------------------------------


def item_time(month, day, hour, mins, now):
    """The datetime of a listed "MM-DD HH:MM": this year, or last year if that lies in the future."""
    for year in (now.year, now.year - 1):
        try:
            at = dt.datetime(year, month, day, hour, mins, tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        if at <= now + dt.timedelta(days=1):
            return at
    return None


def parse_body(body, now):
    """State kept in a digest body: window end, main head, dropped count and the listed items."""
    body = body or ""
    since = SINCE_RE.search(body)
    head = HEAD_RE.search(body)
    dropped = DROPPED_RE.search(body)
    items, section = [], None
    for text in body.splitlines():
        header = HEADER_RE.match(text)
        if header:
            section = SECTION_OF.get(header.group(1))
            continue
        match = LINE_RE.match(text)
        if match and section:
            month, day, hour, mins = (int(g) for g in match.groups()[:4])
            at = item_time(month, day, hour, mins, now)
            if at is not None:
                items.append(item(section, at, match.group(5) or "", match.group(6)))
    return {
        "since": parse_ts(since.group(1)) if since else None,
        "head": head.group(1) if head else None,
        "dropped": (int(dropped.group(1)), dropped.group(2)) if dropped else (0, "-"),
        "items": items,
    }


def merge(old, new, until):
    """Old and new items without duplicates or items older than WINDOW, newest first."""
    cutoff = until - WINDOW
    seen, out = set(), []
    for entry in sorted(new, key=lambda e: e["at"], reverse=True) + list(old):
        ident = key(entry)
        if ident in seen or entry["at"] < cutoff:
            continue
        seen.add(ident)
        out.append(dict(entry, at=minute(entry["at"])))
    out.sort(key=lambda e: e["at"], reverse=True)
    return out


def render(items, run, dropped):
    count, newest = dropped
    total = len(items)
    head = [
        f"Fork-only. `.github/workflows/upstream-digest.yml` rewrites this list every 10 minutes: all "
        f"{UPSTREAM} activity of the last 48 hours, newest first, times in UTC. *own* = {USER}'s own "
        f"item, *bot* = a bot's. It never comments here, so it notifies nobody.",
        "",
        f"**Last run** {run['now'].strftime('%m-%d %H:%M')} UTC · window up to "
        f"{run['until'].strftime('%m-%d %H:%M')} · {total} items · {run['calls']} API calls · "
        f"discussions: {run['discussions']}",
    ]
    if count:
        head.append(f"\n**Size limit:** {count} older items dropped (the newest of them from "
                    f"{newest.replace('T', ' ').rstrip('Z')[5:16]}) to stay under GitHub's "
                    f"{BODY_LIMIT:,}-character limit.")
    out = head
    for section, title in SECTIONS:
        lines = [line(e) for e in items if e["section"] == section]
        out += ["", f"### {title} ({len(lines)})", ""] + lines
    out += ["", f"<!-- digest-since: {iso(run['until'])} -->"]
    if run.get("head"):
        out.append(f"<!-- main-head: {run['head']} -->")
    out.append(f"<!-- dropped: {count} {newest} -->")
    return "\n".join(out) + "\n"


def size(text):
    """Length in UTF-8 bytes: never less than GitHub's character count."""
    return len(text.encode("utf-8"))


def fit(items, run, dropped_before):
    """The body for these items (newest first), dropping the oldest until its size() is under BODY_BUDGET.

    Returns (body, number of items dropped in this run). dropped_before is the (count, newest
    time) carried from earlier runs; it is forgotten once that newest dropped item left the window.
    """
    count, newest = dropped_before
    if newest == "-" or parse_ts(newest) < run["until"] - WINDOW:
        count, newest = 0, "-"
    kept = list(items)
    while True:
        lost = len(items) - len(kept)
        if lost:
            newest_lost = iso(items[len(kept)]["at"])
            mark = max(newest, newest_lost) if newest != "-" else newest_lost
        else:
            mark = newest
        body = render(kept, run, (count + lost, mark))
        excess = size(body) - BODY_BUDGET
        if excess <= 0 or not kept:
            return body, lost
        while excess > 0 and kept:
            excess -= size(line(kept.pop())) + 1


# ---- collection -----------------------------------------------------------------------


class Collector:
    def __init__(self, api, since, until):
        self.api, self.since, self.until = api, since, until
        self.items = []
        self.titles = {}
        self.updated_prs = []
        self.lookups = 0

    def threads(self):
        params = {"state": "all", "sort": "updated", "direction": "desc", "since": iso(self.since)}
        for issue in self.api.pages(f"/repos/{UPSTREAM}/issues", params):
            self.titles[issue["number"]] = issue["title"]
            self.items += thread_events(issue, self.since, self.until)
            if "pull_request" in issue:
                self.updated_prs.append(issue["number"])

    def title(self, number):
        if number not in self.titles:
            if self.lookups >= TITLE_LOOKUPS_PER_RUN:
                return ""
            self.lookups += 1
            self.titles[number] = self.api.get(f"/repos/{UPSTREAM}/issues/{number}")["title"]
        return self.titles[number]

    def comments(self):
        params = {"since": iso(self.since), "sort": "created", "direction": "asc"}
        for c in self.api.pages(f"/repos/{UPSTREAM}/issues/comments", params):
            if in_window(c["created_at"], self.since, self.until):
                number = int(c["issue_url"].rsplit("/", 1)[1])
                self.items.append(comment_item(c, "comment", number, self.title(number)))
        for c in self.api.pages(f"/repos/{UPSTREAM}/pulls/comments", params):
            if in_window(c["created_at"], self.since, self.until):
                number = int(c["pull_request_url"].rsplit("/", 1)[1])
                where = c["path"].rsplit("/", 1)[-1]
                spot = c.get("line") or c.get("original_line")
                kind = f"inline {safe(where, 40)}:{spot}" if spot else f"inline {safe(where, 40)}"
                self.items.append(comment_item(c, kind, number, self.title(number)))

    def reviews(self):
        prs = self.updated_prs[:REVIEW_PRS_PER_RUN]
        if len(self.updated_prs) > len(prs):
            print(f"::notice::reviews read for the {len(prs)} most recently updated of "
                  f"{len(self.updated_prs)} pull requests")
        for number in prs:
            for review in self.api.tail(f"/repos/{UPSTREAM}/pulls/{number}/reviews", pages=1):
                if not in_window(review.get("submitted_at"), self.since, self.until):
                    continue
                if review["state"] == "COMMENTED" and not (review.get("body") or "").strip():
                    continue  # holder of inline comments, those are listed one by one
                kind = REVIEW_KINDS.get(review["state"], "reviewed")
                self.items.append(comment_item(review, kind, number, self.title(number),
                                               review["submitted_at"]))

    def commits(self, head):
        """Commits that reached main since `head` (or, without one, made in the window); returns the new head."""
        commits = None
        if head:
            try:
                data = self.api.get(f"/repos/{UPSTREAM}/compare/{head}...main")
                commits = list(reversed(data.get("commits") or []))
                if data.get("ahead_by", 0) > len(commits):
                    print(f"::warning::main moved {data['ahead_by']} commits, {len(commits)} listed")
            except RuntimeError as err:
                print(f"::warning::compare from {head[:7]} failed, listing by commit date: {err}")
        if commits is None:
            commits = list(self.api.pages(f"/repos/{UPSTREAM}/commits", {"sha": "main", "since": iso(self.since)}))
        self.items += [commit_item(c, self.since, self.until) for c in commits]
        if commits:
            return commits[0]["sha"]
        if head:
            return head
        return self.api.get(f"/repos/{UPSTREAM}/git/ref/heads/main")["object"]["sha"]

    def discussions(self):
        """Discussion items; returns the word for the header ("listed" or why not)."""
        owner, repo = UPSTREAM.split("/", 1)
        try:
            data = self.api.request("POST", "/graphql", body={
                "query": DISCUSSIONS_QUERY, "variables": {"owner": owner, "name": repo}})[0] or {}
        except RuntimeError as err:
            print(f"::warning::discussions not readable: {err}")
            return f"not readable with this token ({safe(str(err), 90)}), skipped"
        if data.get("errors"):
            message = data["errors"][0].get("message", "error")
            print(f"::warning::discussions not readable: {message}")
            return f"not readable with this token ({safe(message, 90)}), skipped"
        repo_data = (data.get("data") or {}).get("repository") or {}
        if not repo_data.get("hasDiscussionsEnabled"):
            return "off upstream"
        self.items += discussion_items(repo_data["discussions"]["nodes"], self.since, self.until)
        return "listed"


# ---- issue ----------------------------------------------------------------------------


def find_digest(api):
    def pick(issues):
        found = [i for i in issues if i["title"] == DIGEST_TITLE and "pull_request" not in i]
        found.sort(key=lambda i: (i["state"] != "open", i["number"]))
        return found[0] if found else None

    issue = pick(api.pages(f"/repos/{FORK}/issues", {"labels": DIGEST_LABEL, "state": "all"}))
    if issue is None:
        issue = pick(api.pages(f"/repos/{FORK}/issues", {"state": "all"}))
    return issue


def create_digest(api, body):
    try:
        api.request("POST", f"/repos/{FORK}/labels", body={
            "name": DIGEST_LABEL, "color": "1d76db", "description": "Fork-only rolling upstream digest"})
    except RuntimeError as err:
        if "HTTP 422" not in str(err):
            raise
    issue = api.request("POST", f"/repos/{FORK}/issues", body={
        "title": DIGEST_TITLE, "body": body, "labels": [DIGEST_LABEL]})[0]
    print(f"created digest issue #{issue['number']}")


def main():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN is not set")
    api = Api(token)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    until = now - SETTLE
    try:
        issue = find_digest(api)
        state = parse_body(issue["body"] if issue else "", now)
        since = max(state["since"] or until - WINDOW, until - WINDOW)
        since = min(since, until)
        print(f"digest #{issue['number'] if issue else '-'}, window {iso(since)} .. {iso(until)}, "
              f"{len(state['items'])} items listed")
        collect = Collector(api, since, until)
        collect.threads()
        collect.comments()
        collect.reviews()
        head = collect.commits(state["head"])
        discussions = collect.discussions()
    except RateLow as err:
        print(f"::warning::stopped early, body kept: {err}")
        return
    items = merge(state["items"], collect.items, until)
    writes = 1 if issue else 2
    run = {"now": now, "until": until, "head": head, "discussions": discussions, "calls": api.calls + writes}
    body, lost = fit(items, run, state["dropped"])
    if DRY_RUN:
        print(body)
    elif issue is None:
        create_digest(api, body)
    else:
        api.request("PATCH", f"/repos/{FORK}/issues/{issue['number']}", body={"body": body})
    summary_line = (f"{len(collect.items)} new items, {len(items) - lost} listed, {lost} dropped for size, "
                    f"{size(body)} bytes, {api.calls} API calls, rate limit remaining {api.remaining}")
    print(summary_line)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(f"Upstream digest: {summary_line}\n")


if __name__ == "__main__":
    main()
