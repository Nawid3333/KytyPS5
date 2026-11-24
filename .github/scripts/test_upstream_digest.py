#!/usr/bin/env python3
"""Tests for upstream_digest.py's pure functions (run: python3 -I .github/scripts/test_upstream_digest.py)."""

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import upstream_digest as ud  # noqa: E402

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 11, 24, 6, 30, 0, tzinfo=UTC)
UNTIL = NOW - ud.SETTLE
SINCE = UNTIL - dt.timedelta(minutes=10)
URL = "https://github.com/KytyPS5/KytyPS5"


def at(text):
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def run(**extra):
    return dict({"now": NOW, "until": UNTIL, "head": "a" * 40, "discussions": "listed", "calls": 7}, **extra)


def issue(number, created, state="open", closed=None, pr=None, user="someone", reason=None):
    data = {"number": number, "title": f"title {number}", "html_url": f"{URL}/issues/{number}",
            "user": {"login": user, "type": "User"}, "created_at": created, "state": state,
            "closed_at": closed, "state_reason": reason}
    if pr is not None:
        data["pull_request"] = pr
        data["html_url"] = f"{URL}/pull/{number}"
    return data


class Tags(unittest.TestCase):
    def test_own_bot_and_plain(self):
        self.assertEqual(ud.tag_of({"login": "nawid3333", "type": "User"}), "own")
        self.assertEqual(ud.tag_of({"login": "coderabbitai[bot]", "type": "Bot"}), "bot")
        self.assertEqual(ud.tag_of({"login": "coderabbitai", "__typename": "Bot"}), "bot")
        self.assertEqual(ud.tag_of({"login": "github-actions", "type": "User"}), "bot")
        self.assertEqual(ud.tag_of({"login": "Crispy81", "type": "User"}), "")
        self.assertEqual(ud.tag_of(None), "")


class ThreadEvents(unittest.TestCase):
    def test_opened_and_merged_in_window_are_two_items(self):
        pr = issue(2535, "2026-11-24T06:21:00Z", "closed", "2026-11-24T06:25:00Z",
                   pr={"merged_at": "2026-11-24T06:25:00Z"}, user="Nawid3333")
        events = ud.thread_events(pr, SINCE, UNTIL)
        self.assertEqual([(e["section"], e["tag"], e["text"].split(" ")[0]) for e in events],
                         [("prs", "own", "opened"), ("prs", "", "merged")])
        self.assertIn("(https://www.github.com/KytyPS5/KytyPS5/pull/2535)", events[0]["text"])
        self.assertNotEqual(ud.key(events[0]), ud.key(events[1]))

    def test_closed_kinds(self):
        unmerged = issue(1, "2026-11-01T00:00:00Z", "closed", "2026-11-24T06:22:00Z", pr={"merged_at": None})
        not_planned = issue(2, "2026-11-01T00:00:00Z", "closed", "2026-11-24T06:22:00Z", reason="not_planned")
        done = issue(3, "2026-11-01T00:00:00Z", "closed", "2026-11-24T06:22:00Z", reason="completed")
        texts = [ud.thread_events(i, SINCE, UNTIL)[0]["text"] for i in (unmerged, not_planned, done)]
        self.assertTrue(texts[0].startswith("closed unmerged [#1]"))
        self.assertTrue(texts[1].startswith("closed as not planned [#2]"))
        self.assertTrue(texts[2].startswith("closed [#3]"))
        self.assertEqual(ud.thread_events(done, SINCE, UNTIL)[0]["section"], "issues")

    def test_outside_the_window_and_reopened_give_nothing(self):
        self.assertEqual(ud.thread_events(issue(4, "2026-11-24T06:00:00Z"), SINCE, UNTIL), [])
        self.assertEqual(ud.thread_events(issue(5, "2026-11-24T06:29:30Z"), SINCE, UNTIL), [])
        reopened = issue(6, "2026-11-01T00:00:00Z", "open", "2026-11-24T06:22:00Z")
        self.assertEqual(ud.thread_events(reopened, SINCE, UNTIL), [])


class Comments(unittest.TestCase):
    def test_excerpt_is_one_safe_line_of_at_most_120_chars(self):
        body = "@Crispy81 see #123 and <!-- digest-since: x -->\n\n" + "word " * 60
        comment = {"user": {"login": "Nmzik", "type": "User"}, "created_at": "2026-11-24T06:22:00Z",
                   "html_url": f"{URL}/issues/2308#issuecomment-1", "body": body}
        entry = ud.comment_item(comment, "comment", 2308, "[GAME STATUS]: Valhalla")
        excerpt = entry["text"].split("Valhalla: ", 1)[1]
        self.assertLessEqual(len(excerpt.replace("​", "").replace("\\", "")), ud.EXCERPT_LEN)
        self.assertNotIn("@Crispy81", entry["text"])
        self.assertNotIn("#123", entry["text"])
        self.assertNotIn("<!--", entry["text"])
        self.assertIn("\\[GAME STATUS\\]", entry["text"])
        self.assertTrue(entry["text"].startswith("Nmzik comment [#2308](https://www.github.com/"))

    def test_commit_time_is_kept_inside_the_window(self):
        commit = {"sha": "4e90654506c8078649e2c1d22c763164406497da", "html_url": f"{URL}/commit/4e90654",
                  "author": None, "commit": {"committer": {"date": "2026-11-20T00:00:00Z"},
                                             "author": {"name": "nmzik"}, "message": "kernel: export rename\n\nbody"}}
        entry = ud.commit_item(commit, SINCE, UNTIL)
        self.assertEqual(entry["at"], SINCE)
        self.assertEqual(entry["text"], f"[4e90654](https://www.github.com/KytyPS5/KytyPS5/commit/4e90654) "
                                        f"nmzik: kernel: export rename")
        commit["commit"]["committer"]["date"] = "2026-11-24T06:29:50Z"
        self.assertLess(ud.commit_item(commit, SINCE, UNTIL)["at"], UNTIL)


class Discussions(unittest.TestCase):
    NODES = [
        {"number": 2529, "title": "Astro", "url": f"{URL}/discussions/2529", "createdAt": "2026-11-24T06:21:00Z",
         "updatedAt": "2026-11-24T06:24:00Z", "author": {"login": "wz11wz", "__typename": "User"},
         "category": {"name": "Q&A"},
         "comments": {"nodes": [
             {"url": f"{URL}/discussions/2529#discussioncomment-1", "createdAt": "2026-11-24T06:23:00Z",
              "body": "first", "author": {"login": "Nawid3333", "__typename": "User"},
              "replies": {"nodes": [
                  {"url": f"{URL}/discussions/2529#discussioncomment-2", "createdAt": "2026-11-24T06:24:00Z",
                   "body": "reply", "author": None},
                  {"url": f"{URL}/discussions/2529#discussioncomment-0", "createdAt": "2026-11-24T05:00:00Z",
                   "body": "old reply", "author": None}]}}]}},
        {"number": 1937, "title": "old", "url": f"{URL}/discussions/1937", "createdAt": "2026-10-27T13:37:26Z",
         "updatedAt": "2026-11-19T22:11:14Z", "author": None, "category": None,
         "comments": {"nodes": [{"url": "x", "createdAt": "2026-11-24T06:23:00Z", "body": "bad clock",
                                 "author": None, "replies": None}]}},
    ]

    def test_new_discussion_comment_and_reply_in_window_only(self):
        items = ud.discussion_items(self.NODES, SINCE, UNTIL)
        self.assertEqual([(e["tag"], e["text"].split(" [")[0]) for e in items],
                         [("", "opened"), ("own", "Nawid3333 comment"), ("", "ghost reply")])
        self.assertTrue(items[0]["text"].endswith("— by wz11wz (Q&A)"))
        self.assertTrue(all(e["section"] == "discussions" for e in items))


class Body(unittest.TestCase):
    def sample(self):
        return [
            ud.item("comments", "2026-11-24T06:25:10Z", "bot", f"coderabbitai reviewed [#2535]({URL}/pull/2535#r1) t"),
            ud.item("prs", "2026-11-24T05:59:00Z", "own", f"opened [#2535]({URL}/pull/2535) t — by Nawid3333"),
            ud.item("commits", "2026-11-23T18:39:00Z", "", f"[4e90654]({URL}/commit/4e9) Nmzik: kernel"),
            ud.item("discussions", "2026-11-23T08:01:00Z", "", f"opened [#2529]({URL}/discussions/2529) A — by w"),
            ud.item("issues", "2026-11-22T08:38:00Z", "", f"opened [#2507]({URL}/issues/2507) 1 — by a"),
        ]

    def test_render_then_parse_gives_the_same_state(self):
        items = ud.merge([], self.sample(), UNTIL)
        body = ud.render(items, run(), (3, "2026-11-22T07:00:00Z"))
        state = ud.parse_body(body, NOW)
        self.assertEqual(state["since"], UNTIL.replace(microsecond=0))
        self.assertEqual(state["head"], "a" * 40)
        self.assertEqual(state["dropped"], (3, "2026-11-22T07:00:00Z"))
        order = [section for section, _ in ud.SECTIONS]
        by_section = sorted(items, key=lambda e: order.index(e["section"]))
        self.assertEqual([(e["section"], e["at"], e["tag"], e["text"]) for e in state["items"]],
                         [(e["section"], e["at"], e["tag"], e["text"]) for e in by_section])
        self.assertIn("### Comments and reviews (1)", body)
        self.assertIn("- 11-24 06:25 *bot* coderabbitai reviewed", body)
        self.assertIn("**Size limit:** 3 older items dropped", body)

    def test_empty_body_is_a_first_run(self):
        self.assertEqual(ud.parse_body("", NOW), {"since": None, "head": None, "dropped": (0, "-"), "items": []})

    def test_lines_outside_known_sections_are_ignored(self):
        body = "- 11-24 06:00 stray\n### Unknown\n- 11-24 06:00 also stray\n### Issues (1)\n- 11-24 06:00 kept\n"
        self.assertEqual([e["text"] for e in ud.parse_body(body, NOW)["items"]], ["kept"])

    def test_year_of_a_listed_time(self):
        new_year = dt.datetime(2027, 1, 1, 0, 5, tzinfo=UTC)
        self.assertEqual(ud.item_time(12, 31, 23, 50, new_year).year, 2026)
        self.assertEqual(ud.item_time(1, 1, 0, 4, new_year).year, 2027)
        self.assertEqual(ud.item_time(2, 29, 10, 0, dt.datetime(2029, 3, 1, tzinfo=UTC)).year, 2028)

    def test_merge_drops_duplicates_and_old_items_newest_first(self):
        old = ud.parse_body(ud.render(ud.merge([], self.sample(), UNTIL), run(), (0, "-")), NOW)["items"]
        same = dict(self.sample()[0])
        later = ud.item("comments", "2026-11-24T06:25:40Z", "", f"Crispy81 comment [#2308]({URL}/issues/2308#c9) t")
        merged = ud.merge(old, [same, later], UNTIL + dt.timedelta(hours=24))
        self.assertEqual(len([e for e in merged if e["section"] == "comments"]), 2)
        self.assertEqual(merged[0]["text"].split(" ")[0], "Crispy81")
        self.assertNotIn("issues", {e["section"] for e in merged})
        self.assertEqual([e["at"] for e in merged], sorted((e["at"] for e in merged), reverse=True))

    def test_merge_keeps_the_given_order_of_items_in_the_same_minute(self):
        first = ud.item("commits", "2026-11-24T06:21:00Z", "", f"[b]({URL}/commit/b) x: newer")
        second = ud.item("commits", "2026-11-24T06:21:00Z", "", f"[a]({URL}/commit/a) x: older")
        self.assertEqual([e["text"] for e in ud.merge([], [first, second], UNTIL)],
                         [first["text"], second["text"]])


class Fit(unittest.TestCase):
    def items(self, count):
        start = UNTIL - dt.timedelta(minutes=1)
        return [ud.item("comments", start - dt.timedelta(minutes=i), "",
                        f"user{i} comment [#{i}]({URL}/issues/{i}#c{i}) t: " + "x" * 100) for i in range(count)]

    def test_everything_fits_under_the_budget(self):
        body, lost = ud.fit(self.items(20), run(), (0, "-"))
        self.assertEqual(lost, 0)
        self.assertIn("<!-- dropped: 0 - -->", body)

    def fit_under(self, items, budget):
        saved = ud.BODY_BUDGET
        ud.BODY_BUDGET = budget
        try:
            return ud.fit(items, run(), (0, "-"))
        finally:
            ud.BODY_BUDGET = saved

    def test_oldest_items_are_dropped_and_counted(self):
        items = self.items(40)
        budget = ud.size(ud.render(items[:25], run(), (99, "2026-11-24T06:00:00Z")))
        body, lost = self.fit_under(items, budget)
        self.assertLessEqual(ud.size(body), budget)
        kept = ud.parse_body(body, NOW)["items"]
        self.assertEqual([e["text"] for e in kept], [e["text"] for e in items[:len(kept)]])
        self.assertEqual(lost, 40 - len(kept))
        self.assertGreater(lost, 0)
        self.assertEqual(ud.parse_body(body, NOW)["dropped"], (lost, ud.iso(items[len(kept)]["at"])))
        self.assertIn(f"**Size limit:** {lost} older items dropped", body)

    def test_the_budget_counts_utf8_bytes(self):
        items = [dict(e, text=e["text"].replace("x" * 100, "游" * 100)) for e in self.items(40)]
        budget = len(ud.render(items[:30], run(), (99, "2026-11-24T06:00:00Z")))
        body, lost = self.fit_under(items, budget)
        self.assertLessEqual(len(body.encode("utf-8")), budget)
        self.assertGreater(lost, 10)

    def test_dropped_count_carries_until_its_items_leave_the_window(self):
        recent = ud.iso(UNTIL - dt.timedelta(hours=40))
        body, _ = ud.fit(self.items(2), run(), (5, recent))
        self.assertEqual(ud.parse_body(body, NOW)["dropped"], (5, recent))
        stale = ud.iso(UNTIL - dt.timedelta(hours=49))
        body, _ = ud.fit(self.items(2), run(), (5, stale))
        self.assertEqual(ud.parse_body(body, NOW)["dropped"], (0, "-"))
        self.assertNotIn("Size limit", body)


if __name__ == "__main__":
    unittest.main()
