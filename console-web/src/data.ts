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
}

export interface ComparisonRow {
  jdId: string;
  kendallTau: number | null;
  deltaU: number | null;
  faithfulness: number | null;
}

interface RealData {
  roles: Role[];
  candidates: Applicant[];
  assessments: Record<string, Assessment>;
  comparison: Record<string, ComparisonRow>;
}

const data = realData as unknown as RealData;

export const ROLES: Role[] = data.roles;
export const APPLICANTS: Applicant[] = data.candidates;
export const COMPARISON: Record<string, ComparisonRow> = data.comparison;

const EMPTY_ASSESSMENT: Assessment = {
  overall_fit_score: 0,
  overall_recommendation: 'no',
  meets_min_qualifications: false,
  requirement_scores: {},
};

export function assessmentFor(applicantRowId: string): Assessment {
  return data.assessments[applicantRowId] ?? EMPTY_ASSESSMENT;
}
