/** Copyright (c) Facebook, Inc. and its affiliates. */
import React from "react";
import Grid from "@material-ui/core/Grid";
import Box from "@material-ui/core/Box";
import Autocomplete from "@material-ui/lab/Autocomplete";
import * as Vega from "react-vega";
import {
  FormControl,
  FormControlLabel,
  FormLabel,
  LinearProgress,
  Radio,
  RadioGroup,
  TextField,
  Typography,
} from "@material-ui/core";
import { Submission, MetricType } from "./data_types";
import { DataGrid } from "@material-ui/data-grid";

type StatsProps = {
  submissions: Submission[];
  submission_lookup: Map<string, Submission>;
};
type StatsState = {
  plot: any;
  error: any;
  initialized: boolean;
  selected_submission: Submission;
  stats_tests: StatisticalTest[];
  selected_test: TestType;
};

type StatisticalTest = {
  model_a: string;
  model_b: string;
  key: string;
  score_a: number;
  score_b: number;
  statistic: number;
  pvalue: number;
  test: string;
  max_score: number;
  min_score: number;
  diff: number;
  fold: string;
  metric: string;
};

enum TestType {
  Wilcoxon = "wilcoxon",
  StudentT = "student_t",
  McNemar = "mcnemar",
  SEM = "sem",
  SEE = "see",
}

export class Stats extends React.Component<StatsProps, StatsState> {
  state: StatsState = {
    plot: null,
    error: null,
    initialized: false,
    selected_submission: this.props.submissions[0],
    stats_tests: [],
    selected_test: TestType.SEM,
  };
  componentDidMount() {
    fetch("/api/1.0/stats/plot")
      .then((res) => res.json())
      .then(
        (result) => {
          const vega_plot = <Vega.VegaLite spec={result} />;
          this.setState({
            plot: vega_plot,
            initialized: true,
          });
        },
        (error) => {
          this.setState({
            error: error,
          });
        }
      );
  }
  updateSelectedSubmission(submission: Submission | null) {
    if (submission === null) {
      return;
    }
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        submission_id: submission.submission_id,
      }),
    };
    fetch("/api/1.0/stats/by-model", requestOptions)
      .then((res) => res.json())
      .then(
        (result) => {
          this.setState({
            stats_tests: result["tests"] as StatisticalTest[],
          });
        },
        (error) => {
          this.setState({
            error: error,
          });
        }
      );
  }
  updateSelectedTest(test_type: TestType) {
    this.setState({ selected_test: test_type });
  }
  renderTestControls() {
    const test_options = [];
    for (let test in TestType) {
      const test_typed = test as keyof typeof TestType;
      test_options.push(
        <FormControlLabel
          key={test}
          value={TestType[test_typed]}
          control={<Radio />}
          label={TestType[test_typed]}
        />
      );
    }
    return (
      <FormControl component="fieldset" style={{ marginTop: "20px" }}>
        <FormLabel component="legend">Choose Test Type</FormLabel>
        <RadioGroup
          aria-label="plot_type"
          name="plot_type"
          value={this.state.selected_test}
          onChange={(e) => this.updateSelectedTest(e.target.value as TestType)}
          row
        >
          {test_options}
        </RadioGroup>
      </FormControl>
    );
  }
  renderTestsTable() {
    const test_rows = [];
    var idx = 1;
    const columns = [
      { field: "rank", headerName: "Rank", width: 70 },
      { field: "name", headerName: "Name", width: 200 },
      { field: "diff", headerName: "Diff", width: 110 },
      { field: "pvalue", headerName: "P-Value", width: 110 },
      { field: "score", headerName: "Other Score", width: 110 },
      { field: "test", headerName: "Test" },
    ];
    for (let row of this.state.stats_tests) {
      if (row.test !== this.state.selected_test) {
        continue;
      }
      const name =
        this.state.selected_submission.submission_id === row.model_a
          ? row.model_b
          : row.model_a;
      const other_score =
        this.state.selected_submission.submission_id === row.model_a
          ? row.score_b
          : row.score_a;
      test_rows.push({
        id: row.key,
        rank: idx,
        pvalue: row.pvalue.toFixed(4),
        diff_number: row.diff,
        diff: row.diff.toFixed(4),
        name: name,
        score: other_score.toFixed(4),
        test: row.test,
        fold: row.fold,
        metric: row.metric,
      });
      idx += 1;
    }
    const sorted_tests = test_rows.sort(
      (a, b) => a.diff_number - b.diff_number
    );
    return (
      <DataGrid
        rows={sorted_tests}
        columns={columns}
        sortingOrder={["desc", "asc", null]}
      />
    );
  }
  renderSubmissionTable() {
    const submission_rows = [];
    var idx = 1;
    const columns = [
      { field: "rank", headerName: "Rank", width: 70 },
      { field: "name", headerName: "Name", width: 200 },
      { field: "devEM", headerName: "Dev EM", width: 110 },
      { field: "testEM", headerName: "Test EM", width: 110 },
      {
        field: "diffEM",
        headerName: "Diff EM",
        type: "number",
        cellClassName: "",
        width: 120,
      },
    ];
    for (let row of this.props.submissions) {
      if (row.submission_id === this.state.selected_submission.submission_id) {
        continue;
      }
      submission_rows.push({
        id: row.submission_id,
        rank: idx,
        name: row.name,
        submission_id: row.submission_id,
        bundle_id: row.bundle_id,
        devEM: (100 * row.dev_scores[MetricType.EM]).toPrecision(3),
        devF1: (100 * row.dev_scores[MetricType.F1]).toPrecision(3),
        devSkill: row.dev_skill.toPrecision(3),
        testEM: row.test_scores[MetricType.EM].toPrecision(3),
        testF1: row.test_scores[MetricType.F1].toPrecision(3),
        diffEM: (
          row.test_scores[MetricType.EM] -
          100 * row.dev_scores[MetricType.EM]
        ).toPrecision(3),
        diffF1: (
          row.test_scores[MetricType.F1] -
          100 * row.dev_scores[MetricType.F1]
        ).toPrecision(3),
      });
      idx += 1;
    }
    return (
      <DataGrid
        rows={submission_rows}
        columns={columns}
        sortingOrder={["desc", "asc", null]}
      />
    );
  }
  render() {
    var content = this.state.initialized ? (
      this.state.plot
    ) : (
      <LinearProgress></LinearProgress>
    );
    return (
      <Box style={{ padding: "30px" }}>
        <Grid container spacing={3} justify="center">
          <Grid item xs={11} spacing={4}>
            <Typography variant="h3">
              Comparing Statistical Tests of Significance
            </Typography>
            <Typography>
              This plot compares four tests of statistical significance. For all
              pairs of SQuAD models, the P-Value for their exact match scores
              was computed using these tests. The plot shows for tests resulting
              in a P-Value of greater than .05, what the corresponding
              difference in exact match score was. The compared tests are :
              McNemar's, Student T-Test, Wilcoxon, and Statistical Error of
              Measurement (from IRT).
            </Typography>
          </Grid>
          <Grid item xs={11} justify="center" spacing={4}>
            {content}
          </Grid>
          <Grid item xs={8} justify="center">
            <Typography variant="h3" style={{ marginBottom: "20px" }}>
              Compare Significance of a Model
            </Typography>
            <Autocomplete
              options={this.props.submissions}
              getOptionLabel={(option: Submission) => option.name}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Select Model"
                  variant="outlined"
                />
              )}
              onChange={(event, newValue) =>
                this.updateSelectedSubmission(newValue)
              }
            ></Autocomplete>
            {this.renderTestControls()}
          </Grid>
          <Grid item xs={11} style={{ minHeight: "600px" }}>
            {this.renderTestsTable()}
          </Grid>
        </Grid>
      </Box>
    );
  }
}

export default Stats;
