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
  utility: number;
}

export interface Assessment {
  overall_fit_score: number;
  overall_recommendation: 'hire' | 'maybe' | 'no';
  meets_min_qualifications: boolean;
  requirement_scores: Record<string, number>;
  seniority_fit_score: number | null;
  education_fit_score: number | null;
}

export interface ComparisonRow {
  jdId: string;
  meanFitScore: number | null;
  meetsMinRate: number | null;
  hireRate: number | null;
  rankingStability: number | null;
}

export interface EvaluationSummary {
  nPairs: number | null;
  nRepeats: number | null;
  recommendationAgreementRate: number | null;
  overallScoreStdev: number | null;
  meanRankingConvergence: number | null;
  coherenceSpearmanRho: number | null;
}

export interface ShortlistingFlaggedApplicant {
  candidateId: string;
  overallFitScore: number;
  nearZeroRequirements: string[];
}

export interface ShortlistingAuditRow {
  nCandidates: number;
  nFlagged: number;
  flaggedPoolFraction: number;
  profileFlagged: boolean;
  flaggedCandidates: ShortlistingFlaggedApplicant[];
}

interface RealData {
  roles: Role[];
  candidates: Applicant[];
  assessments: Record<string, Assessment>;
  comparison: Record<string, ComparisonRow>;
  evaluationSummary: EvaluationSummary | null;
  shortlistingAudit: Record<string, ShortlistingAuditRow> | null;
}

const data = realData as unknown as RealData;

export const ROLES: Role[] = data.roles;
export const APPLICANTS: Applicant[] = data.candidates;
export const COMPARISON: Record<string, ComparisonRow> = data.comparison;
export const EVALUATION_SUMMARY: EvaluationSummary | null = data.evaluationSummary ?? null;
export const SHORTLISTING_AUDIT: Record<string, ShortlistingAuditRow> = data.shortlistingAudit ?? {};

const EMPTY_ASSESSMENT: Assessment = {
  overall_fit_score: 0,
  overall_recommendation: 'no',
  meets_min_qualifications: false,
  requirement_scores: {},
  seniority_fit_score: null,
  education_fit_score: null,
};

export function assessmentFor(applicantRowId: string): Assessment {
  return data.assessments[applicantRowId] ?? EMPTY_ASSESSMENT;
}
