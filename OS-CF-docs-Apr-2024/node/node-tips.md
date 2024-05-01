# Node.js buildpack additional information
You can use node-specific information to supplement the general guidelines in
the [Pushing an app](https://docs.cloudfoundry.org/devguide/deploy-apps/deploy-app.html) topic.
For information about using and extending the Node.js buildpack in Cloud
Foundry, see the [nodejs-buildpack repository](https://github.com/cloudfoundry/nodejs-buildpack) in GitHub.
You can find current information about this buildpack on the Node.js buildpack [release page](https://github.com/cloudfoundry/nodejs-buildpack/releases) in GitHub.
The buildpack uses a default Node.js version.
To specify the versions of Node.js and npm an app requires, edit the app’s `package.json`, as described in “node.js and npm versions” in the [nodejs-buildpack repository](https://github.com/cloudfoundry/nodejs-buildpack).

## Application package file
Cloud Foundry expects a `package.json` in your Node.js app.
You can specify the version of Node.js you want to use in the `engine` node of
your `package.json` file.
In general, Cloud Foundry supports the two most recent versions of Node.js.
See the GitHub [Node.js buildpack page](https://github.com/cloudfoundry/nodejs-buildpack/releases) for current information.
Example `package.json` file:
```
{
"name": "first",
"version": "0.0.1",
"author": "Demo",
"dependencies": {
"express": "3.4.8",
"consolidate": "0.10.0",
"swig": "1.3.2"
},
"engines": {
"node": "0.12.7",
"npm": "2.7.4"
}
}
```

## Application port
You must use the PORT environment variable to determine which port your
app listens on. To also run your app locally, set the default port as `3000`.
```
app.listen(process.env.PORT || 3000);
```

## Low Memory environments
When running node apps, you might notice that instances are occasionally
restarted due to memory constraints. Node does not know how much memory it is
allowed to use, and thus sometimes allows the garbage collector to wait past
the allowed amount of memory. To resolve this issue, set the `OPTIMIZE_MEMORY` environment variable to `true` (requires node v6.12.0 or greater). This sets `max_old_space_size` based on the available memory in the instance.
```
$ cf set-env my-app OPTIMIZE_MEMORY true
```

## Application start command
Node.js apps require a start command.
You can specify the web start command for a Node.js app in a Procfile or in the app deployment manifest. For more information about Procfiles, see the [Configuring a Production Server](https://docs.cloudfoundry.org/buildpacks/prod-server.html) topic.
The first time you deploy, you are asked if you want to save your configuration.
This saves a `manifest.yml` in your app with the settings you
entered during the initial push.
Edit the `manifest.yml` file and create a start command as follows:
```

---
applications:

- name: my-app
command: node my-app.js
... the rest of your settings ...
```
Alternately, specify the start command with `cf push -c`.
```
$ cf push my-app -c "node my-app.js"
```

## Application bundling
You do not need to run `npm install` before deploying your app.
Cloud Foundry runs it for you when your app is pushed.
You can, if you prefer, run `npm install` and create a `node_modules` folder
inside of your app.

## Solve discovery problems
If Cloud Foundry does not automatically detect that your app is a Node.js app, you can override auto-detection by specifying the Node.js buildpack.
Add the buildpack into your `manifest.yml` and re-run `cf push` with your
manifest:
```

---
applications:

- name: my-app
buildpacks: https://github.com/cloudfoundry/nodejs-buildpack
... the rest of your settings ...
```
Alternately, specify the buildpack on the command line with `cf push -b`:
```
$ cf push my-app -b https://github.com/cloudfoundry/nodejs-buildpack
```

## Bind services
Refer to [Configure Service Connections for Node.js](https://docs.cloudfoundry.org/buildpacks/node/node-service-bindings.html).

## Environment variables
You can access environments variable programmatically.
For example, you can obtain `VCAP_SERVICES` as follows:
```
process.env.VCAP_SERVICES
```
Environment variables available to you include both those [defined by the system](https://docs.cloudfoundry.org/devguide/deploy-apps/environment-variable.html#app-system-env)
and those defined by the Node.js buildpack, as described below.

### BUILD\_DIR directory
Directory into which Node.js is copied each time a Node.js app is run.

### CACHE\_DIR directory
Directory that Node.js uses for caching.

### PATH
The system path used by Node.js.
`PATH=/home/vcap/app/bin:/home/vcap/app/node_modules/.bin:/bin:/usr/bin`