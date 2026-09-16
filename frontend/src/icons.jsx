/**
 * Inline SVG icons (stroke-based, 24-unit grid) so the UI has no icon-font
 * or image dependency. Each takes an optional `size` and passes through
 * className/style.
 */
const base = (size) => ({
  width: size, height: size, viewBox: "0 0 24 24", fill: "none",
  stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round",
});

const make = (paths) => ({ size = 20, ...rest }) => (
  <svg {...base(size)} {...rest} aria-hidden="true">{paths}</svg>
);

export const IconPencil = make(<>
  <path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
</>);
export const IconSearch = make(<>
  <circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" />
</>);
export const IconSidebar = make(<>
  <rect x="3" y="4" width="18" height="16" rx="3" /><path d="M9 4v16" />
</>);
export const IconBook = make(<>
  <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5Z" /><path d="M4 20.5V5.5" /><path d="M8 7h8" />
</>);
export const IconCode = make(<>
  <path d="m8 8-4 4 4 4" /><path d="m16 8 4 4-4 4" /><path d="m14 4-4 16" />
</>);
export const IconImage = make(<>
  <rect x="3" y="4" width="18" height="16" rx="3" /><circle cx="9" cy="10" r="1.6" /><path d="m21 16-5-5-9 9" />
</>);
export const IconMic = make(<>
  <rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0" /><path d="M12 18v3" />
</>);
export const IconShield = make(<>
  <path d="M12 3 4.5 6v5.5c0 4.6 3.2 8 7.5 9.5 4.3-1.5 7.5-4.9 7.5-9.5V6Z" /><path d="m9 12 2 2 4-4" />
</>);
export const IconSliders = make(<>
  <path d="M4 7h10" /><path d="M18 7h2" /><circle cx="16" cy="7" r="2" />
  <path d="M4 17h4" /><path d="M12 17h8" /><circle cx="10" cy="17" r="2" />
</>);
export const IconPlus = make(<><path d="M12 5v14" /><path d="M5 12h14" /></>);
export const IconArrowUp = make(<><path d="M12 19V5" /><path d="m6 11 6-6 6 6" /></>);
export const IconStop = make(<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none" />);
export const IconChevronDown = make(<path d="m6 9 6 6 6-6" />);
export const IconChevronRight = make(<path d="m9 6 6 6-6 6" />);
export const IconUpDown = make(<><path d="m7 9 5-5 5 5" /><path d="m7 15 5 5 5-5" /></>);
export const IconTrash = make(<>
  <path d="M4 7h16" /><path d="M10 11v6" /><path d="M14 11v6" /><path d="M6 7l1 13h10l1-13" /><path d="M9 7V4h6v3" />
</>);
export const IconDots = make(<>
  <circle cx="5" cy="12" r="1.4" fill="currentColor" /><circle cx="12" cy="12" r="1.4" fill="currentColor" /><circle cx="19" cy="12" r="1.4" fill="currentColor" />
</>);
export const IconPaperclip = make(
  <path d="m21 11.5-8.5 8.5a5 5 0 0 1-7-7l9-9a3.3 3.3 0 0 1 4.7 4.7l-9 9a1.6 1.6 0 0 1-2.3-2.3L16 7.5" />
);
export const IconCheck = make(<path d="m5 12 4 4L19 6" />);
export const IconCopy = make(<>
  <rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V6a2 2 0 0 1 2-2h9" />
</>);
export const IconClock = make(<><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" /></>);
export const IconSparkle = make(
  <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8Z" fill="currentColor" stroke="none" />
);
export const IconDownload = make(<><path d="M12 4v11" /><path d="m7 10 5 5 5-5" /><path d="M5 20h14" /></>);
export const IconX = make(<><path d="M6 6l12 12" /><path d="M18 6 6 18" /></>);
export const IconFile = make(<>
  <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z" /><path d="M14 3v5h5" />
</>);

/** Brand mark: a hexagonal node — “sovereign perimeter”. */
export function Logo({ size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2.5 20.5 7v10L12 21.5 3.5 17V7Z" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
      <circle cx="12" cy="12" r="3" fill="currentColor" />
    </svg>
  );
}

export const IconLogout = make(<>
  <path d="M10 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h4" /><path d="m15 8 5 4-5 4" /><path d="M20 12H9" />
</>);
