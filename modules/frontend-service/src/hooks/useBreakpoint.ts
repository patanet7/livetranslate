import { useState, useEffect } from "react";
import { useTheme } from "@mui/material/styles";
import useMediaQuery from "@mui/material/useMediaQuery";

type Breakpoint = "xs" | "sm" | "md" | "lg" | "xl";

export const useBreakpoint = (): Breakpoint => {
  const theme = useTheme();

  const isXs = useMediaQuery(theme.breakpoints.only("xs"));
  const isSm = useMediaQuery(theme.breakpoints.only("sm"));
  const isMd = useMediaQuery(theme.breakpoints.only("md"));
  const isLg = useMediaQuery(theme.breakpoints.only("lg"));
  const isXl = useMediaQuery(theme.breakpoints.only("xl"));

  const [breakpoint, setBreakpoint] = useState<Breakpoint>("lg");

  useEffect(() => {
    if (isXs) setBreakpoint("xs");
    else if (isSm) setBreakpoint("sm");
    else if (isMd) setBreakpoint("md");
    else if (isLg) setBreakpoint("lg");
    else if (isXl) setBreakpoint("xl");
  }, [isXs, isSm, isMd, isLg, isXl]);

  return breakpoint;
};

export const useIsMobile = (): boolean => {
  const breakpoint = useBreakpoint();
  return breakpoint === "xs" || breakpoint === "sm";
};

export const useIsTablet = (): boolean => {
  const breakpoint = useBreakpoint();
  return breakpoint === "md";
};

export const useIsDesktop = (): boolean => {
  const breakpoint = useBreakpoint();
  return breakpoint === "lg" || breakpoint === "xl";
};
