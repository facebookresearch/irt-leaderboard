/** Copyright (c) Facebook, Inc. and its affiliates. */
import React from "react";
import CssBaseline from "@material-ui/core/CssBaseline";
import AppBar from "@material-ui/core/AppBar";
import LinearProgress from "@material-ui/core/LinearProgress";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Container from "@material-ui/core/Container";
import Paper from "@material-ui/core/Paper";
import Button from "@material-ui/core/Button";
import "./App.css";
import { Submission } from "./data_types";
import Leaderboard from "./Leaderboard";
import Stats from "./Stats";
import Paired from "./Paired";
import InspectExamples from "./InspectExamples";
import IrtExamples from "./IrtExamples";
import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link as RouterLink,
} from "react-router-dom";

type AppProps = {};
type AppState = {
  submissions: Submission[];
  submission_lookup: Map<string, Submission>;
  initialized: boolean;
  error: any;
};

var navButtonStyle = { marginLeft: "15px" };

export class App extends React.Component<AppProps, AppState> {
  state: AppState = {
    submissions: [],
    submission_lookup: new Map(),
    error: null,
    initialized: false,
  };
  componentDidMount() {
    fetch("/api/1.0/submissions")
      .then((res) => res.json())
      .then(
        (result) => {
          const submissions = [];
          const submission_lookup = new Map();
          for (let s of result["submissions"]) {
            const parsed_submission = {
              bundle_id: s["bundle_id"],
              created: new Date(Date.parse(s["created"])),
              name: s["name"],
              dev_scores: s["dev_scores"],
              test_scores: s["test_scores"],
              submission_id: s["submission_id"],
              submitter: s["submitter"],
              task: s["task"],
              dev_skill: s["dev_skill"],
            };
            submission_lookup.set(
              parsed_submission.submission_id,
              parsed_submission
            );
            submissions.push(parsed_submission);
          }
          this.setState({
            submissions: submissions,
            submission_lookup: submission_lookup,
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
  render() {
    var leaderboard = <LinearProgress />;
    var pairwise = <LinearProgress />;
    var examples = <LinearProgress />;
    var irt = <LinearProgress></LinearProgress>;
    var stats = <LinearProgress></LinearProgress>;
    if (this.state.initialized) {
      leaderboard = (
        <Leaderboard
          submissions={this.state.submissions}
          submission_lookup={this.state.submission_lookup}
        />
      );
      pairwise = (
        <Paired
          submissions={this.state.submissions}
          submission_lookup={this.state.submission_lookup}
        />
      );
      examples = (
        <InspectExamples
          submissions={this.state.submissions}
          submission_lookup={this.state.submission_lookup}
        />
      );
      irt = <IrtExamples></IrtExamples>;
      stats = (
        <Stats
          submissions={this.state.submissions}
          submission_lookup={this.state.submission_lookup}
        />
      );
    }
    return (
      <Router>
        <div>
          <CssBaseline />
          <AppBar position="static">
            <Toolbar>
              <div
                style={{
                  backgroundColor: "white",
                  width: "50px",
                  height: "50px",
                  borderRadius: "50%",
                  marginRight: "10px",
                }}
              >
                <img
                  src="pedroai-logo.png"
                  alt="Website logo"
                  style={{ marginTop: "5px" }}
                />
              </div>
              <Typography component="h1" variant="h6" color="inherit" noWrap>
                SQuAD 2.0 Leaderboard
              </Typography>
              <Button
                color="primary"
                variant="contained"
                component={RouterLink}
                style={navButtonStyle}
                disableElevation
                to="/"
              >
                Main Page
              </Button>
              <Button
                component={RouterLink}
                to="/pairwise"
                style={navButtonStyle}
                variant="contained"
                disableElevation
                color="primary"
              >
                Pairwise Model Comparison
              </Button>
              <Button
                component={RouterLink}
                to="/examples"
                style={navButtonStyle}
                variant="contained"
                disableElevation
                color="primary"
              >
                Example Visualization
              </Button>
              <Button
                component={RouterLink}
                to="/irt"
                style={navButtonStyle}
                variant="contained"
                disableElevation
                color="primary"
              >
                Item Response Theory
              </Button>
              <Button
                component={RouterLink}
                to="/stats"
                style={navButtonStyle}
                variant="contained"
                disableElevation
                color="primary"
              >
                Statistical Test Comparison
              </Button>
              <Button
                component={RouterLink}
                to="/jake"
                style={navButtonStyle}
                variant="contained"
                disableElevation
                color="primary"
              >
                STAPLE Parallel Coordinate
              </Button>
            </Toolbar>
          </AppBar>
          <Container maxWidth="lg" fixed>
            <Paper elevation={3} style={{ paddingTop: "20px" }}>
              <Switch>
                <Route path="/examples">{examples}</Route>
                <Route path="/stats">{stats}</Route>
                <Route path="/pairwise">{pairwise}</Route>
                <Route path="/irt">{irt}</Route>
                <Route path="/">{leaderboard}</Route>
              </Switch>
            </Paper>
          </Container>
        </div>
      </Router>
    );
  }
}

export default App;
