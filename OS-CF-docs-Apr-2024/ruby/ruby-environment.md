# Environment variables in Ruby buildpack
Cloud Foundry provides configuration information to apps through environment variables.
You can use additional environment variables that are provided by the Ruby buildpack.
For more information about the standard environment variables,
see [Cloud Foundry Environment Variables](https://docs.cloudfoundry.org/devguide/deploy-apps/environment-variable.html).

## Ruby buildpack environment variables
The following table describes the environment variables provided by the Ruby buildpack:
| Environment Variable | Description |
| --- | --- |
| `BUNDLE_BIN_PATH` | The directory where Bundler installs binaries. For example: `BUNDLE_BIN_PATH:/home/vcap/app/vendor/bundle/ruby/1.9.1/gems/bundler-1.3.2/bin/bundle` |
| `BUNDLE_GEMFILE` | The path to the Gemfile for the app. For example: `BUNDLE_GEMFILE:/home/vcap/app/Gemfile` |
| `BUNDLE_WITHOUT` | Instructs Cloud Foundry to skip gem installation in excluded groups. Use this with Rails apps, where “assets” and “development” gem groups typically contain gems that are not needed when the app runs in production. For example: `BUNDLE_WITHOUT=assets` |
| `DATABASE_URL` | Cloud Foundry examines the `database_uri` for bound services to see if they match known database types. If known relational database services are bound to the app, then the `DATABASE_URL` environment variable is set to the first services in the list. If your app requires that `DATABASE_URL` is set to the connection string for your service, and Cloud Foundry does not set it, use the Cloud Foundry Command Line Interface (cf CLI) `cf set-env` command to set this variable manually. For example: `cf set-env my-app DATABASE_URL mysql://example-database-connection-string` |
| `GEM_HOME` | The directory where gems are installed. For example: `GEM_HOME:/home/vcap/app/vendor/bundle/ruby/1.9.1` |
| `GEM_PATH` | The directory where gems can be found. For example: `GEM_PATH=/home/vcap/app/vendor/bundle/ruby/1.9.1:` |
| `RACK_ENV` | The Rack deployment environment, which governs the middleware loaded to run the app. Valid value are `development`, `deployment`, and `none`. For example: `RACK_ENV=none` |
| `RAILS_ENV` | The Rails deployment environment, which controls which environment-specific configuration file governs how the app is run. Valid values are `development`, `test`, and `production`. For example: `RAILS_ENV=production` |
| `RUBYOPT` | Defines command-line options passed to Ruby interpreter. For example: `RUBYOPT: -I/home/vcap/app/vendor/bundle/ruby/1.9.1/gems/bundler-1.3.2/lib -rbundler/setup` |