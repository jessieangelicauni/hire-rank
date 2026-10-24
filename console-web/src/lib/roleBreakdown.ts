import { APPLICANTS, assessmentFor, type AssessmentDetail, type Role } from '../data';

export interface RoleBreakdown {
  requirementMeans: [string, number][];
}

function mean(values: number[]): number | null {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

export function roleBreakdownFor(role: Role): RoleBreakdown {
  const assessments: AssessmentDetail[] = APPLICANTS
    .filter((a) => a.roleId === role.id)
    .map((a) => assessmentFor(a.id));

  const requirementValues = new Map<string, number[]>();
  for (const a of assessments) {
    for (const item of a.breakdown) {
      const list = requirementValues.get(item.requirement) ?? [];
      list.push(item.expectedStrength);
      requirementValues.set(item.requirement, list);
    }
  }
  const requirementMeans: [string, number][] = Array.from(requirementValues.entries())
    .map(([requirement, values]): [string, number] => [requirement, mean(values) ?? 0])
    .sort(([, a], [, b]) => b - a);

  return { requirementMeans };
}
