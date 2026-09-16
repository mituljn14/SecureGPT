/**
 * Minimal markdown -> React renderer for agent answers.
 *
 * Covers what the agents actually emit (headings, bullet / numbered lists,
 * **bold**, *italic*, `code`, fenced code blocks, simple pipe tables) and
 * nothing else. Builds React nodes directly -- no HTML string is ever
 * injected, so model output cannot smuggle markup into the page.
 */
import { Fragment } from "react";

const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*|\[[^\]]+\]\([^)\s]+\))/g;

function inline(text, keyBase = "i") {
  const parts = text.split(INLINE).filter((p) => p !== "");
  return parts.map((p, i) => {
    const k = `${keyBase}-${i}`;
    if (p.startsWith("**") && p.endsWith("**")) return <strong key={k}>{p.slice(2, -2)}</strong>;
    if (p.startsWith("`") && p.endsWith("`")) return <code key={k}>{p.slice(1, -1)}</code>;
    if (p.startsWith("*") && p.endsWith("*") && p.length > 2) return <em key={k}>{p.slice(1, -1)}</em>;
    const link = p.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/);
    if (link) {
      return <a key={k} href={link[2]} target="_blank" rel="noreferrer">{link[1]}</a>;
    }
    return <Fragment key={k}>{p}</Fragment>;
  });
}

function table(lines, key) {
  const rows = lines
    .filter((l) => !/^\s*\|?\s*:?-{2,}/.test(l))
    .map((l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim()));
  if (!rows.length) return null;
  const [head, ...body] = rows;
  return (
    <div className="md-table-wrap" key={key}>
      <table className="md-table">
        <thead><tr>{head.map((c, i) => <th key={i}>{inline(c, `${key}h${i}`)}</th>)}</tr></thead>
        <tbody>
          {body.map((r, ri) => (
            <tr key={ri}>{r.map((c, ci) => <td key={ci}>{inline(c, `${key}r${ri}c${ci}`)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Markdown({ text = "" }) {
  const lines = String(text).replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;
  let k = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim().startsWith("```")) {
      const buf = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) buf.push(lines[i++]);
      i += 1;
      out.push(<pre className="md-code" key={k++}><code>{buf.join("\n")}</code></pre>);
      continue;
    }

    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      const Tag = `h${Math.min(h[1].length + 2, 5)}`;
      out.push(<Tag className="md-h" key={k++}>{inline(h[2], `h${k}`)}</Tag>);
      i += 1;
      continue;
    }

    if (/^\s*\|.*\|\s*$/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) buf.push(lines[i++]);
      const t = table(buf, `t${k++}`);
      if (t) out.push(t);
      continue;
    }

    if (/^\s*[-*•]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*•]\s+/, ""));
        i += 1;
      }
      out.push(<ul className="md-list" key={k++}>{items.map((t, j) => <li key={j}>{inline(t, `u${k}${j}`)}</li>)}</ul>);
      continue;
    }

    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+[.)]\s+/, ""));
        i += 1;
      }
      out.push(<ol className="md-list" key={k++}>{items.map((t, j) => <li key={j}>{inline(t, `o${k}${j}`)}</li>)}</ol>);
      continue;
    }

    if (line.trim() === "") { i += 1; continue; }

    const buf = [line];
    i += 1;
    while (i < lines.length && lines[i].trim() !== "" && !/^(#{1,4}\s|\s*[-*•]\s|\s*\d+[.)]\s|```|\s*\|)/.test(lines[i])) {
      buf.push(lines[i++]);
    }
    out.push(
      <p className="md-p" key={k++}>
        {buf.map((l, j) => <Fragment key={j}>{j > 0 && <br />}{inline(l, `p${k}${j}`)}</Fragment>)}
      </p>
    );
  }

  return <div className="md">{out}</div>;
}
