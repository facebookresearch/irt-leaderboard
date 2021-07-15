/** Copyright (c) Facebook, Inc. and its affiliates. */
import React from "react";
import Alert from "@material-ui/lab/Alert";
import * as Vega from "react-vega";
import TextField from "@material-ui/core/TextField";
import Autocomplete from "@material-ui/lab/Autocomplete";
import { DataGrid } from "@material-ui/data-grid";
import { LinearProgress, Typography } from "@material-ui/core";
import Box from "@material-ui/core/Box";
import Grid from "@material-ui/core/Grid";
import { Example, Submission } from "./data_types";

type InspectExamplesProps = {
  submissions: Submission[];
  submission_lookup: Map<string, Submission>;
};
type InspectExamplesState = {
  examples: Example[];
  initialized: boolean;
  error: any;
  selected_submissions: Submission[];
  plot: any;
};

export class InspectExamples extends React.Component<
  InspectExamplesProps,
  InspectExamplesState
> {
  state: InspectExamplesState = {
    initialized: false,
    examples: [],
    error: null,
    plot: null,
    selected_submissions: [],
  };
  componentDidMount() {
    fetch("/api/1.0/examples")
      .then((res) => res.json())
      .then(
        (result) => {
          this.setState({
            examples: result["examples"],
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
  updateSubmissions(submissions: Submission[]) {
    if (submissions.length > 0) {
      const requestOptions = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          submission_ids: Array.from(submissions).map((s) => s.submission_id),
        }),
      };
      fetch("/api/1.0/examples/plot", requestOptions)
        .then((res) => res.json())
        .then(
          (result) => {
            const vega_plot = <Vega.VegaLite spec={result} />;
            this.setState({
              plot: vega_plot,
              selected_submissions: submissions,
            });
          },
          (error) => {
            this.setState({
              error: error,
            });
          }
        );
    } else {
      this.setState({ selected_submissions: [] });
    }
  }

  render() {
    var examples = <LinearProgress></LinearProgress>;
    var plot = null;
    if (this.state.initialized) {
      const columns = [
        { field: "example_id", headerName: "ID", width: 100 },
        { field: "article", headerName: "Article", width: 100 },
        { field: "text", headerName: "Text", width: 600 },
        { field: "disc", headerName: "Discriminability", width: 100 },
        { field: "diff", headerName: "Difficulty", width: 100 },
      ];
      const rows = Array.from(this.state.examples).map((e) => {
        return {
          id: e.example_id,
          article: e.data.article,
          example_id: e.example_id,
          text: e.data.question,
          disc: e.disc_pyro_2pl,
          diff: e.diff_pyro_2pl,
        };
      });
      examples = (
        <DataGrid rows={rows} columns={columns} pageSize={7}></DataGrid>
      );
      plot = (
        <Alert severity="warning">
          No models selected, cannot create a plot
        </Alert>
      );
      if (this.state.plot !== null) {
        plot = this.state.plot;
      }
    }
    return (
      <Grid container justify="center" spacing={1}>
        <Grid item xs={12} style={{ margin: "20px" }}>
          <Typography variant="h3">Visualize Examples</Typography>
          <Typography>
            This visualization helps users explore how model performance varies
            with the types of examples they select. Choose any number of models
            with the dropdown.
          </Typography>
          <Autocomplete
            multiple
            options={this.props.submissions}
            getOptionLabel={(option: Submission) => option.name}
            renderInput={(params) => (
              <TextField {...params} label="Select Models" variant="outlined" />
            )}
            onChange={(event, newValue) => this.updateSubmissions(newValue)}
          ></Autocomplete>
          <Box style={{ marginTop: "30px" }}>{plot}</Box>
        </Grid>
        <Grid item xs={11}>
          <Box style={{ minHeight: "490px" }}>{examples}</Box>
        </Grid>
      </Grid>
    );
  }
}

export default InspectExamples;
