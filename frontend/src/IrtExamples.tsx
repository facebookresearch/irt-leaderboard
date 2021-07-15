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

type IrtExamplesProps = {};
type IrtExamplesState = {
  examples: Example[];
  initialized: boolean;
  error: any;
  plot: any;
};

export class IrtExamples extends React.Component<
  IrtExamplesProps,
  IrtExamplesState
> {
  state: IrtExamplesState = {
    initialized: false,
    examples: [],
    error: null,
    plot: null,
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
    fetch("/api/1.0/examples/plot_irt")
      .then((res) => res.json())
      .then(
        (result) => {
          const vega_plot = <Vega.VegaLite spec={result} />;
          this.setState({
            plot: vega_plot,
          });
        },
        (error) => {
          this.setState({
            error: error,
          });
        }
      );
  }

  render() {
    var loading = true;
    var examples = null;
    var plot = null;
    if (this.state.initialized) {
      loading = false;
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
      if (this.state.plot !== null) {
        plot = this.state.plot;
      }
    }
    var content = loading ? <LinearProgress></LinearProgress> : plot;
    return (
      <Grid container justify="center" spacing={1}>
        <Grid item xs={12} style={{ margin: "20px" }}>
          <Typography variant="h3">Visualize Examples</Typography>
          <Box style={{ marginTop: "30px" }}>{content}</Box>
        </Grid>
        {/* <Grid item xs={11}>
          <Box style={{ minHeight: "490px" }}>{examples}</Box>
        </Grid> */}
      </Grid>
    );
  }
}

export default IrtExamples;
