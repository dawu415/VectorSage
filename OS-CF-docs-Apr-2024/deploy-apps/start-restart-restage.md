# Starting, restarting, and restaging your apps
You can start, restart, and restage apps in Cloud Foundry, using the cf CLI or the app manifest (attributes in a YAML file).

## Start your app
To start your app, run the following command from your app root directory:
```
$ cf push YOUR-APP
```
For more information about pushing apps, see [Pushing an app](https://docs.cloudfoundry.org/devguide/deploy-apps/deploy-app.html).
Cloud Foundry determines the start command for your app from one of these three sources:

* The `-c` command-line option in the Cloud Foundry Command Line Interface (cf CLI). For example:
```
$ cf push YOUR-APP -c "node YOUR-APP.js"
```

* The `command` attribute in the app manifest. For example:
```
command: node YOUR-APP.js
```

* The buildpack, which provides a start command appropriate for a particular type of app.
The source that Cloud Foundry uses depends on factors that are explained in the next section.

### How Cloud Foundry determines its default start command
The first time you deploy an app, `cf push` uses the buildpack start command by default.
After that, `cf push` defaults to whatever start command was used for the previous push.
To override these defaults, provide the `-c` option, or the command attribute in the manifest.
When you provide start commands both at the command line and in the manifest, `cf push` ignores the command in the manifest.

### Forcing Cloud Foundry to use the buildpack start command
To force Cloud Foundry to use the buildpack start command, specify a start command of `null`.
You can specify a null start command in one of two ways.

* Using the `-c` command-line option in the cf CLI:
```
$ cf push YOUR-APP -c "null"
```

* Using the `command` attribute in the app manifest:
```
command: null
```
This can be helpful after you have deployed while providing a start command at the command line or the manifest.
Now a command that you provided, rather than the buildpack start command, has become the default start command.
In this situation, if you decide to deploy using the buildpack start command, the `null` command simplifies that.

### Start commands when migrating a database
Start commands are used in special ways when you migrate a database as part of an app deployment. For more information, see [Services overview](https://docs.cloudfoundry.org/devguide/services/#migrating).

## Restart your app
To restart your app, run:
```
$ cf restart YOUR-APP
```
Restarting your app stops your app and restarts it with the already compiled droplet. A droplet is a tarball that includes:

* stack

* [buildpack](https://docs.cloudfoundry.org/buildpacks/)

* app source code
The Diego [cell](https://docs.cloudfoundry.org/concepts/architecture/#diego-cell) unpacks, compiles, and runs a droplet on a container.
Restart your app to refresh the app’s environment after actions such as binding a new service to the app or setting an environment variable that only the app consumes. However, if your environment variable is consumed by the buildpack in addition to the app, then you must [restage](https://docs.cloudfoundry.org/devguide/deploy-apps/start-restart-restage.html#restage) the app for the change to take effect.

## Restage your app
To restage your app, run:
```
$ cf restage YOUR-APP
```
Restaging your app stops your app and restages it, by compiling a new droplet and starting it.
Restage your app if you have changed the environment in a way that affects your staging process, such as setting an environment variable that the buildpack consumes. Staging has access to environment variables, so the environment can affect the contents of the droplet. You must also restage your app whenever you edit any configuration settings, such as when you rename it, add metadata, or configure health checks. The new settings often do not take effect until you restage the app.
Restaging your app compiles a new droplet from your app without updating your app source. If you must update your app source, re-push your app by following the steps in [Start your app](https://docs.cloudfoundry.org/devguide/deploy-apps/start-restart-restage.html#start).