import { useEffect, useState } from "react";

/** Tracks viewport width crossing a breakpoint. Default: 640px (Tailwind `sm`). */
export function useIsMobile(breakpoint = 640): boolean {
  const get = () =>
    typeof window === "undefined" ? false : window.innerWidth < breakpoint;
  const [mobile, setMobile] = useState<boolean>(get);

  useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${breakpoint - 1}px)`);
    const onChange = () => setMobile(mq.matches);
    onChange();
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [breakpoint]);

  return mobile;
}
