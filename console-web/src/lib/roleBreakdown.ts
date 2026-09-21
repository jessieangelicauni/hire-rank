import { APPLICANTS, assessmentFor, type Assessment, type Role } from '../data';

export interface RoleBreakdown {
  seniorityMean: number | null;
  educationMean: number | null;
  recommendationCounts: { hire: number; maybe: number; no: number };
  requirementMeans: [string, number][];
}

function mean(values: number[]): number | null {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

export function roleBreakdownFor(role: Role): RoleBreakdown {
  const assessments: Assessment[] = APPLICANTS
    .filter((a) => a.roleId === role.id)
    .map((a) => assessmentFor(a.id));

  const seniorityValues = assessments
    .map((a) => a.seniority_years_fit_score)
    .filter((v): v is number => v !== null);
  const educationValues = assessments
    .map((a) => a.education_fit_score)
    .filter((v): v is number => v !== null);

  const recommendationCounts = { hire: 0, maybe: 0, no: 0 };
  for (const a of assessments) recommendationCounts[a.overall_recommendation] += 1;

  const requirementValues = new Map<string, number[]>();
  for (const a of assessments) {
    for (const [requirement, score] of Object.entries(a.requirement_scores)) {
      const list = requirementValues.get(requirement) ?? [];
      list.push(score);
      requirementValues.set(requirement, list);
    }
  }
  const requirementMeans: [string, number][] = Array.from(requirementValues.entries())
    .map(([requirement, values]): [string, number] => [requirement, mean(values) ?? 0])
    .sort(([, a], [, b]) => b - a);

  return {
    seniorityMean: mean(seniorityValues),
    educationMean: mean(educationValues),
    recommendationCounts,
    requirementMeans,
  };
}
