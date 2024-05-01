# Updating buildpack-related gems in Cloud Foundry
Learn how to update your [buildpack-packager](https://github.com/cloudfoundry/buildpack-packager) and [machete](https://github.com/cloudfoundry/machete) CF buildpack test framework, which are used for CF system buildpack development.
The `buildpack-packager` packages buildpacks and `machete` provides an integration test framework.
The CF Buildpacks team uses the [gems-and-extensions pipeline](https://buildpacks.ci.cf-app.com/teams/main/pipelines/gems-and-extensions) to:

* Run integration tests for `buildpack-packager` and `machete`.

* Update the gems in the buildpacks managed by the team.

## Running the update process
The following steps assume you are using a Concourse deployment of the `buildpacks-ci` pipelines.
At the end of the process, there is a new GitHub release. Updates are then applied to the buildpacks.
To update the version of either gem in a buildpack:

1. Verify that the test job `<gemname>-specs` for the gem was updated successfully and ran on the commit you plan to update.

2. Start the `<gemname>-tag` job to update (“bump”) the version of the gem.
The `<gemname>-release` job starts and creates a new GitHub release of the gem.

3. Each of the buildpack pipelines, for example, the [go-buildpack pipeline](https://buildpacks.ci.cf-app.com/teams/main/pipelines/go-buildpack) has a job that watches for new releases of the gem. When a new release is detected, the buildpack’s `cf.Gemfile` is updated to that release version.

4. The commit made to the buildpack’s `cf.Gemfile` starts the full integration test suite for that buildpack.
The final step starts all buildpack test suites simultaneously,
and causes contention for available shared BOSH lite test environments.