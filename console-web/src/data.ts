import realData from './data/real-data.json';

export interface Role {
  id: string;
  title: string;
  description: string;
  candidateCount: number;
}

export interface Applicant {
  id: string;
  cvId: string;
  roleId: string;
  name: string;
  initials: string;
  rank: number | null;
  utility: number | null;
  criterionScore: number | null;
}

export interface CriterionBreakdownItem {
  requirement: string;
  evidenceStrength: 'strong' | 'partial' | 'none';
  expectedStrength: number;
  probabilities: Record<string, number> | null;
}

export interface SkillGroup {
  names: string[];
  context: string;
}

export interface AssessmentDetail {
  criteriaMet: number;
  criteriaTotal: number;
  criterionScore: number;
  breakdown: CriterionBreakdownItem[];
  skillsGrouped: SkillGroup[];
}

export interface ComparisonRow {
  jdId: string;
  totalAssessed: number | null;
  meanCriterionScore: number | null;
}

export interface ConfidenceBucket {
  mean: number;
  n: number;
}

export interface ReliabilitySummary {
  /** Does Jev's holistic pairwise judgment agree with the transparent
   * criterion_score on clearly-separated pairs? (concurrent validity) */
  tieringValidity: { total: number; agreed: number; agreementRate: number | null } | null;
  /** Does higher reported confidence correlate with higher self-agreement
   * across repeats? (is confidence a meaningful signal) */
  confidenceCalibration: { buckets: Record<string, ConfidenceBucket> } | null;
  /** Does the pairwise winner stay stable across repeats, and across
   * reordered (cosmetic) evidence presentation? */
  rankingConsistency: {
    totalPairs: number;
    flippedPairs: number;
    inconsistentRate: number;
    cosmeticTotal: number;
    cosmeticFlipped: number;
    cosmeticInconsistentRate: number;
  } | null;
  /** Does the full ranking tournament reproduce the same order across
   * independent runs? (Kendall's tau) */
  rankingRepeatability: {
    nRepeatPairs: number;
    meanKendallTau: number | null;
    /** How many candidates were actually re-tested live via Jev (inside a
     * tie cluster) vs. assigned a deterministic score with zero live calls.
     * A tau of 1.0 with nCandidatesInTieClusters === 0 is trivial, not proof
     * of stability — nothing was actually re-queried. */
    nCandidates?: number | null;
    nCandidatesInTieClusters?: number | null;
    nJevCalls?: number | null;
  } | null;
  coverage?: string;
}

interface RealData {
  roles: Role[];
  candidates: Applicant[];
  assessments: Record<string, AssessmentDetail>;
  comparison: Record<string, ComparisonRow>;
  reliability?: Record<string, ReliabilitySummary>;
}

const data = realData as RealData;

export const ROLES: Role[] = data.roles;
export const APPLICANTS: Applicant[] = data.candidates;
export const COMPARISON: Record<string, ComparisonRow> = data.comparison;
const RELIABILITY: Record<string, ReliabilitySummary> = data.reliability ?? {};

export function reliabilityFor(roleId: string): ReliabilitySummary | null {
  return RELIABILITY[roleId] ?? null;
}

const EMPTY_ASSESSMENT: AssessmentDetail = {
  criteriaMet: 0,
  criteriaTotal: 0,
  criterionScore: 0,
  breakdown: [],
  skillsGrouped: [],
};

export function assessmentFor(applicantRowId: string): AssessmentDetail {
  return data.assessments[applicantRowId] ?? EMPTY_ASSESSMENT;
}
