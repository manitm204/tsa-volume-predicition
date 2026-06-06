import { gradeClass, type Grade } from "@/lib/signal";
import { cn } from "@/lib/utils";

export function GradePill({
  grade, size = "md", className,
}: {
  grade: Grade;
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  const sz = size === "sm" ? "text-xs h-6 min-w-[28px]"
           : size === "lg" ? "text-2xl h-12 min-w-[56px] rounded-xl"
           : "";
  return (
    <span className={cn(gradeClass(grade), sz, className)}>
      {grade}
    </span>
  );
}
