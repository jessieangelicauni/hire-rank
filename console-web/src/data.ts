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
  composite_fit_score: number;
  requirement_scores: Record<string, number>;
  seniority_years_fit_score: number | null;
  education_fit_score: number | null;
}

export interface RepeatSample {
  compositeFitScore: number;
}

export interface ComparisonRow {
  jdId: string;
  meanFitScore: number | null;
  rankingStability: number | null;
}

export interface EvaluationSummary {
  nPairs: number | null;
  nRepeats: number | null;
  recommendationAgreementRate: number | null;
  compositeScoreStdev: number | null;
  meanRankingConvergence: number | null;
  coherenceSpearmanRho: number | null;
}

interface RealData {
  roles: Role[];
  candidates: Applicant[];
  assessments: Record<string, Assessment>;
  comparison: Record<string, ComparisonRow>;
  evaluationSummary: EvaluationSummary | null;
  repeatSamples: Record<string, RepeatSample[]>;
}

const data = realData as RealData;

export const ROLES: Role[] = data.roles;
export const APPLICANTS: Applicant[] = data.candidates;
export const COMPARISON: Record<string, ComparisonRow> = data.comparison;
export const EVALUATION_SUMMARY: EvaluationSummary | null = data.evaluationSummary ?? null;

const EMPTY_ASSESSMENT: Assessment = {
  composite_fit_score: 0,
  requirement_scores: {},
  seniority_years_fit_score: null,
  education_fit_score: null,
};

export function assessmentFor(applicantRowId: string): Assessment {
  return data.assessments[applicantRowId] ?? EMPTY_ASSESSMENT;
}

export const REPEAT_SAMPLES: Record<string, RepeatSample[]> = data.repeatSamples ?? {};

export function repeatSamplesFor(applicantRowId: string): RepeatSample[] {
  return REPEAT_SAMPLES[applicantRowId] ?? [];
}
