/** Copyright (c) Facebook, Inc. and its affiliates. */
import React from "react";
import TextField from "@material-ui/core/TextField";
import Autocomplete from "@material-ui/lab/Autocomplete";
import Grid from "@material-ui/core/Grid";
import Box from "@material-ui/core/Box";
import * as Vega from "react-vega";
import { Typography } from "@material-ui/core";
import { Submission } from "./data_types";

type PairedProps = {
  submissions: Submission[];
  submission_lookup: Map<string, Submission>;
};
type PairedState = {
  submission_1: Submission;
  submission_2: Submission;
  plot: any;
  error: any;
};

export class Paired extends React.Component<PairedProps, PairedState> {
  state: PairedState = {
    plot: null,
    error: null,
    submission_1: this.props.submissions[0],
    submission_2: this.props.submissions[1],
  };
  componentDidMount() {
    this.updatePlot(this.state.submission_1, this.state.submission_2);
  }
  updatePlot(submission_1: Submission, submission_2: Submission) {
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        submission_id_1: submission_1.submission_id,
        submission_name_1: submission_1.name,
        submission_id_2: submission_2.submission_id,
        submission_name_2: submission_2.name,
      }),
    };
    fetch("/api/1.0/pairwise/plot", requestOptions)
      .then((res) => res.json())
      .then(
        (result) => {
          const vega_plot = <Vega.VegaLite spec={result} />;
          this.setState({
            plot: vega_plot,
            submission_1: submission_1,
            submission_2: submission_2,
          });
        },
        (error) => {
          this.setState({
            error: error,
          });
        }
      );
  }
  updateSubmission1(submission: Submission | null) {
    if (submission !== null) {
      this.updatePlot(submission, this.state.submission_2);
    }
  }
  updateSubmission2(submission: Submission | null) {
    if (submission !== null) {
      this.updatePlot(this.state.submission_1, submission);
    }
  }
  render() {
    return (
      <Box style={{ padding: "30px" }}>
        <Grid container spacing={3} justify="center">
          <Grid item xs={10}>
            <Typography variant="h4">
              Pairwise Model Comparison on Example Attributes
            </Typography>
            <Typography>
              Another useful way to compare models is to see how they differ on
              subsets of the dataset. In SQuAD, questions are based on Wikipedia
              articles so this comparison shows how models compare based on the
              article the questions are from. For ease of comparison, you can
              click on articles in the legend to highlight them across all
              panes, and ctrl-click to select multiple.
            </Typography>
          </Grid>
          <Grid item container xs={12} spacing={2}>
            <Grid item xs={6}>
              <Autocomplete
                options={this.props.submissions}
                defaultValue={this.state.submission_1}
                getOptionLabel={(option: Submission) => option.name}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Select Model 1"
                    variant="outlined"
                  />
                )}
                onChange={(event, newValue) => this.updateSubmission1(newValue)}
              ></Autocomplete>
            </Grid>
            <Grid item xs={6}>
              <Autocomplete
                options={this.props.submissions}
                defaultValue={this.state.submission_2}
                getOptionLabel={(option: Submission) => option.name}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Select Model 2"
                    variant="outlined"
                  />
                )}
                onChange={(event, newValue) => this.updateSubmission2(newValue)}
              ></Autocomplete>
            </Grid>
          </Grid>
          <Grid item xs={11}>
            <Typography>Model 1: {this.state.submission_1.name}</Typography>
            <Typography>Model 2: {this.state.submission_2.name}</Typography>
          </Grid>
          <Grid item xs={12} justify="center">
            {this.state.plot}
          </Grid>
        </Grid>
      </Box>
    );
  }
}

export default Paired;
