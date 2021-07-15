/** Copyright (c) Facebook, Inc. and its affiliates. */
import React from "react";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableContainer from "@material-ui/core/TableContainer";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import Paper from "@material-ui/core/Paper";
import Radio from "@material-ui/core/Radio";
import RadioGroup from "@material-ui/core/RadioGroup";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import FormLabel from "@material-ui/core/FormLabel";
import Button from "@material-ui/core/Button";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import { withStyles } from "@material-ui/core/styles";
import { DataGrid, CellClassParams } from "@material-ui/data-grid";
import Grid from "@material-ui/core/Grid";
import Box from "@material-ui/core/Box";
import FormControl from "@material-ui/core/FormControl";
import * as Vega from "react-vega";
import { Typography } from "@material-ui/core";
import { Submission, MetricType } from "./data_types";

// function encodeQueryData(data: any) {
//   const ret = [];
//   for (let d in data)
//     ret.push(encodeURIComponent(d) + "=" + encodeURIComponent(data[d]));
//   return ret.join("&");
// }

function diffToColor(params: CellClassParams) {
  const diff = params.value as number;
  if (diff > 5) {
    return "positive-color";
  } else if (diff > 0) {
    return "positive-color";
  } else if (-5 < diff) {
    return "neutral-color";
  } else if (-10 < diff) {
    return "caution-color";
  } else {
    return "negative-color";
  }
}

const styles = (theme: any) => ({
  formControl: {
    margin: theme.spacing(1),
    minWidth: 120,
  },
  selectEmpty: {
    marginTop: theme.spacing(2),
  },
});

type LeaderboardProps = {
  theme: any;
  classes: any;
  submissions: Submission[];
  submission_lookup: Map<string, Submission>;
};

type LeaderboardState = {
  selected_submissions: Set<string>;
  timeseries_plot: any;
  metrics_plot: any;
  ranks_plot: any;
  error: string | null;
  sort_metric: Metric;
  parsed_metric: ParsedMetric;
  selected_plot: PlotType;
  generated_plot: any;
};

enum Metric {
  DevEM = "dev_exact_match",
  DevF1 = "dev_f1",
  TestEM = "test_exact_match",
  TestF1 = "test_f1",
  DevSkill = "dev_skill",
}

enum PlotType {
  BarCompare = "Bar Chart Comparison",
  ChordCompare = "Chord Chart Comparison",
  ParallelCoordinate = "Parallel Coordinate Plot",
}

type ParsedMetric = {
  fold: string;
  name: string;
};

function parseMetric(metric: Metric): ParsedMetric {
  const sorting_name = metric.replace("dev_", "").replace("test_", "");
  if (metric.includes("dev")) {
    return { fold: "dev", name: sorting_name };
  } else if (metric.includes("test")) {
    return { fold: "test", name: sorting_name };
  } else {
    throw new Error("Invalid metric");
  }
}

export class Leaderboard extends React.Component<
  LeaderboardProps,
  LeaderboardState
> {
  state: LeaderboardState = {
    selected_submissions: new Set(),
    timeseries_plot: null,
    metrics_plot: null,
    ranks_plot: null,
    error: null,
    sort_metric: Metric.DevEM,
    parsed_metric: parseMetric(Metric.DevEM),
    selected_plot: PlotType.BarCompare,
    generated_plot: null,
  };
  componentDidMount() {
    fetch("/api/1.0/submissions/plot")
      .then((res) => res.json())
      .then(
        (result) => {
          this.setState({
            timeseries_plot: result,
          });
        },
        (error) => {
          this.setState({
            error: error,
          });
        }
      );
    fetch("/api/1.0/metrics/plot")
      .then((res) => res.json())
      .then(
        (result) => {
          this.setState({
            metrics_plot: result,
          });
        },
        (error) => {
          this.setState({
            error: error,
          });
        }
      );
  }

  // updateSortMetric(metric: any) {
  //   this.setState((state) => {
  //     const metric_info = parseMetric(metric);
  //     const sorted_submissions = state.submissions.sort((a, b) => {
  //       // We want descending numbers, so invert signs
  //       var a_score = null;
  //       var b_score = null;
  //       if (metric_info.fold === "dev") {
  //         a_score = -a.dev_scores[metric_info.name];
  //         b_score = -b.dev_scores[metric_info.name];
  //       } else if (metric_info.fold === "test") {
  //         a_score = -a.test_scores[metric_info.name];
  //         b_score = -b.test_scores[metric_info.name];
  //       } else {
  //         throw new Error("Invalid fold");
  //       }
  //       if (a_score < b_score) {
  //         return -1;
  //       } else if (a_score > b_score) {
  //         return 1;
  //       } else {
  //         return 0;
  //       }
  //     });
  //     return {
  //       sort_metric: metric,
  //       parsed_metric: metric_info,
  //       submissions: sorted_submissions,
  //     };
  //   });
  // }

  updateSelectedSubmissions(submission_ids: Array<string>) {
    this.setState((state) => {
      return { selected_submissions: new Set(submission_ids) };
    });
  }

  updateSelectedPlot(plot_type: PlotType) {
    this.setState({
      selected_plot: plot_type,
    });
  }

  // renderSortControls() {
  //   return (
  //     <FormControl className={this.props.classes.formControl}>
  //       <InputLabel id="demo-simple-select-label">Sort Models By</InputLabel>
  //       <Select
  //         labelId="demo-simple-select-label"
  //         id="demo-simple-select"
  //         value={this.state.sort_metric}
  //         onChange={(e) => this.updateSortMetric(e.target.value)}
  //       >
  //         <MenuItem value={Metric.DevEM}>Dev Exact Match (EM)</MenuItem>
  //         <MenuItem value={Metric.DevF1}>Dev F1</MenuItem>
  //         <MenuItem value={Metric.DevSkill}>Dev Skill</MenuItem>
  //         <MenuItem value={Metric.TestEM}>Test Exact Match (EM)h</MenuItem>
  //         <MenuItem value={Metric.TestF1}>Test F1</MenuItem>
  //       </Select>
  //     </FormControl>
  //   );
  // }
  renderPlotControls() {
    const plot_options = [];
    for (let plot in PlotType) {
      const plot_typed = plot as keyof typeof PlotType;
      plot_options.push(
        <FormControlLabel
          key={plot}
          value={PlotType[plot_typed]}
          control={<Radio />}
          label={PlotType[plot_typed]}
        />
      );
    }
    const submission_ids = Array.from(this.state.selected_submissions.values());
    const selected_submissions = [];
    for (let sub_id of submission_ids) {
      console.log(sub_id);
      const sub = this.props.submission_lookup.get(sub_id);
      if (sub != undefined) {
        selected_submissions.push(
          <TableRow key={sub_id}>
            <TableCell component="th" scope="row">
              {sub.name}
            </TableCell>
          </TableRow>
        );
      }
    }
    if (selected_submissions.length === 0) {
      selected_submissions.push(
        <TableRow key="no-models">
          <TableCell component="th" scope="row">
            No Model Submissions Selected
          </TableCell>
        </TableRow>
      );
    }
    return (
      <Card variant="outlined" style={{ marginTop: "30px" }}>
        <CardContent>
          <Typography variant="body1" component="p">
            Instructions: Select models in the table, choose a plot type, and
            click "Plot"
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={5}>
              <FormControl component="fieldset" style={{ marginTop: "20px" }}>
                <FormLabel component="legend">Choose Plot Type</FormLabel>
                <RadioGroup
                  aria-label="plot_type"
                  name="plot_type"
                  value={this.state.selected_plot}
                  onChange={(e) =>
                    this.updateSelectedPlot(e.target.value as PlotType)
                  }
                >
                  {plot_options}
                </RadioGroup>
              </FormControl>
            </Grid>
            <Grid item xs={7}>
              <TableContainer component={Paper}>
                <Table aria-label="model table">
                  <TableHead>
                    <TableRow>
                      <TableCell>Selected Models</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>{selected_submissions}</TableBody>
                </Table>
              </TableContainer>
            </Grid>
            <Grid item xs={12}>
              <Button
                variant="contained"
                color="primary"
                onClick={(_) => this.updateGeneratedPlot()}
              >
                Create Plot
              </Button>
            </Grid>
            <Box>{this.state.generated_plot}</Box>
          </Grid>
        </CardContent>
      </Card>
    );
  }

  updateGeneratedPlot() {
    if (this.state.selected_plot === PlotType.ChordCompare) {
      const plot = (
        <iframe
          src="/chord.html"
          width="980px"
          height="720px"
          frameBorder="0"
        ></iframe>
      );
      this.setState({ generated_plot: plot });
    } else {
      var path = null;
      if (this.state.selected_plot === PlotType.BarCompare) {
        path = "/api/1.0/submissions/plot_compare";
      } else if (this.state.selected_plot === PlotType.ParallelCoordinate) {
        path = "/api/1.0/ranks/plot";
      } else {
        path = "/api/1.0/submissions/plot_compare";
      }
      const requestOptions = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          submission_ids: Array.from(this.state.selected_submissions),
        }),
      };
      fetch(path, requestOptions)
        .then((res) => res.json())
        .then(
          (result) => {
            const vega_plot = <Vega.VegaLite key="generated" spec={result} />;
            this.setState({
              generated_plot: vega_plot,
            });
          },
          (error) => {
            this.setState({
              error: error,
            });
          }
        );
    }
  }

  renderSubmissionTable() {
    const submission_rows = [];
    var idx = 1;
    const columns = [
      { field: "rank", headerName: "Rank", width: 70 },
      { field: "name", headerName: "Name", width: 200 },
      { field: "devEM", headerName: "Dev EM", width: 110 },
      { field: "devF1", headerName: "Dev F1", width: 110 },
      { field: "devSkill", headerName: "Dev Skill", width: 120 },
      { field: "testEM", headerName: "Test EM", width: 110 },
      { field: "testF1", headerName: "Test F1", width: 110 },
      {
        field: "diffEM",
        headerName: "Diff EM",
        type: "number",
        cellClassName: diffToColor,
        width: 120,
      },
      {
        field: "diffF1",
        headerName: "Diff F1",
        type: "number",
        cellClassName: diffToColor,
        width: 120,
      },
    ];
    for (let row of this.props.submissions) {
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
        onSelectionChange={(newSelection) => {
          this.updateSelectedSubmissions(
            newSelection.rowIds.map((k) => k.toString())
          );
        }}
        sortingOrder={["desc", "asc", null]}
        rowsPerPageOptions={[500, 100, 50, 25]}
        checkboxSelection
      />
    );
  }
  render() {
    var vega_plots = [];
    var timeseries_plot = null;
    if (this.state.timeseries_plot != null) {
      timeseries_plot = (
        <Vega.VegaLite key="timeseries" spec={this.state.timeseries_plot} />
      );
    }
    if (this.state.metrics_plot != null) {
      vega_plots.push(
        <Vega.VegaLite key="metrics" spec={this.state.metrics_plot} />
      );
    }
    if (this.state.ranks_plot != null) {
      vega_plots.push(
        <Vega.VegaLite key="ranks" spec={this.state.ranks_plot} />
      );
    }
    //const controls = this.renderSortControls();
    const controls = null;
    const submissionTable = this.renderSubmissionTable();
    const plotControls = this.renderPlotControls();
    return (
      <Grid container spacing={1} justify="center" style={{ padding: "30px" }}>
        <Grid item xs={12}>
          {controls}
        </Grid>
        <Grid item xs={12}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="body1" component="p">
                Instructions: The plot is moveable, scrollable and selecting
                legend elements filters by metric.
              </Typography>
              {timeseries_plot}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} style={{ minHeight: "600px", marginBottom: "60px" }}>
          <Box>
            <Typography>Click on Columns to Sort</Typography>
          </Box>
          {submissionTable}
        </Grid>
        <Grid item xs={12}></Grid>
        <Grid item xs={12}>
          {plotControls}
        </Grid>
      </Grid>
    );
  }
}

export default withStyles(styles, { withTheme: true })(Leaderboard);
