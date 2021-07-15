/** Copyright (c) Facebook, Inc. and its affiliates. */
export type Example = {
  example_id: string;
  task: string;
  data: Data;
  disc_pyro_2pl: number;
  diff_pyro_2pl: number;
};

export type Data = {
  answers?: string[];
  article: string;
  id: string;
  is_impossible: boolean;
  plausible_answers?: any;
  question: string;
};

export type Submission = {
  bundle_id: string;
  created: Date;
  name: string;
  dev_scores: any;
  test_scores: any;
  submission_id: string;
  submitter: string;
  task: string;
  dev_skill: number;
};

export enum MetricType {
  EM = "exact_match",
  F1 = "f1",
}
