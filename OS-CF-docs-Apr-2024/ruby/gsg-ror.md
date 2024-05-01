# Deploying Ruby on Rails apps
You can use this information to deploy a Ruby on Rails app to Cloud Foundry.

## Prerequisites
In order to deploy a sample Ruby on Rails app, you must have the following:

* Cloud Foundry deployment

* [Cloud Foundry Command Line Interface](https://docs.cloudfoundry.org/cf-cli/install-go-cli.html)

* Cloud Foundry username and password with **Space Developer** [permissions](https://docs.cloudfoundry.org/concepts/roles.html#roles). See your [Org Manager](https://docs.cloudfoundry.org/concepts/roles.html#roles) if you require permissions.

## Step 1: Clone the app
Run the following command to create a local copy of the cf-sample-app-rails:
```
git clone https://github.com/cloudfoundry-samples/cf-sample-app-rails.git
```
The newly created directory contains a `manifest.yml` file, which assists CF with deploying the app.
See [Deploying with Application Manifests](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest.html#minimal-manifest) for more information.

## Step 2: Log in and target the API endpoint

1. Run the following terminal command to log in and target the API endpoint of your deployment. For more information, see [Identifying your Cloud Foundry API Endpoint and Version](http://docs.cloudfoundry.org/running/cf-api-endpoint.html).
```
cf login -a YOUR-API-ENDPOINT
```

2. Use your credentials to log in, and to select a [Space and Org](https://docs.cloudfoundry.org/concepts/roles.html).

## Step 3: Create a service instance
Run the following terminal command to create a PostgreSQL service instance for the sample app.
```
cf create-service postgresql-10-odb standalone rails-postgres
```
For example:
```
$ cf create-service postgresql-10-odb standalone rails-postgres
Creating service rails-postgres in org YOUR-ORG / space development as clouduser@example.com....
OK
```
The service instance is `rails-postgres`. It uses the `postgresql-10-odb` service and the `standalone` plan.
For more information about the `postgresql-10-odb` service, see
[Crunchy PostgreSQL](https://www.crunchydata.com/products/crunchy-postgresql-for-cloud-foundry/).

## Step 4: Deploy the app
Make sure you are in the `cf-sample-app-rails` directory. Run the following command to deploy the app:
```
cf push cf-sample-app-rails
```
This command creates a URL route to your application in the form `HOST.DOMAIN`.
In this example, `HOST` is `cf-sample-app-rails`. Administrators specify the `DOMAIN`.
For example, for the `DOMAIN` `shared-domain.example.com`, running the previous command
creates the URL `cf-sample-app-rails.shared-domain.example.com`.
The following example shows the terminal output when deploying the `cf-sample-app-rails`. `cf push` uses the instructions in the manifest file to create the app, create and bind the route, and upload the app. It then follows the information in the manifest to start one instance of the app with 256M of RAM. After the app starts, the output displays the health and status of the app.
```
$ cf push cf-sample-app-rails
Using manifest file ~/workspace/cf-sample-app-rails/manifest.yml
Creating app cf-sample-app-rails in org my-rog / space dev as clouduser@example.com...
OK
Creating route cf-sample-app-rails.cfapps.io...
OK
Binding cf-sample-app-rails.cfapps.io to cf-sample-app-rails...
OK
Uploading cf-sample-app-rails...
Uploading app files from: ~/workspace/cf-sample-app-rails
Uploading 746.6K, 136 files
Done uploading
OK
Starting app cf-sample-app-rails in org my-org / space dev as clouduser@example.com...
. . .
0 of 1 instances running, 1 starting
1 of 1 instances running
App started
OK
App cf-sample-app-rails was started using this command `bundle exec rails server -p $PORT`
Showing health and status for app cf-sample-app-rails in org my-org / space dev as clouduser@example.com...
OK
requested state: started
instances: 1/1
usage: 512M x 1 instances
urls: cf-sample-app-rails.cfapps.io
last uploaded: Fri Dec 22 18:08:32 UTC 2017
stack: cflinuxfs3
buildpack: ruby
state since cpu memory disk details

#0 running 2018-8-17 10:09:57 AM 0.0% 20.7M of 512M 186.8M of 1G
```
If you want to view log activity while the app deploys, launch a new terminal window and
run `cf logs cf-sample-app-rails`.
To avoid security exposure, verify that you migrate your apps and custom buildpacks to use
the `cflinuxfs4` stack based on Ubuntu 22.04 LTS (Jammy Jellyfish). The `cflinuxfs3`
stack is based on Ubuntu 18.04 (Bionic Beaver), which reaches end of standard support in April 2023.

## Step 5: Bind the service instance

1. Run the following command to bind the service instance to the sample app.
Once bound, environment variables are stored that allow the app to connect to the
service after a `cf push`, `cf restage`, or `cf restart` command.
```
$ cf bind-service cf-sample-app-rails rails-postgres
Binding service rails-postgres to app cf-sample-app-rails in org my-org / space dev
OK
TIP: Use 'cf restage cf-sample-app-rails' to ensure your env variable changes take effect
```

2. Run the following command to restage the sample app.
```
$ cf restage cf-sample-app-rails
```

3. Run the following command to verify the service instance is bound to the sample app.
```
$ cf services
Getting services in org my-org / space dev
OK
name service plan bound apps last operation
rails-postgres postgresql-10-odb standalone cf-sample-app-rails create succeeded
```

## Step 6: Verify the app
Verify that your app is running by browsing to the URL generated in the output of the previous step.
In this example, go to `cf-sample-app-rails.shared-domain.example.com` and verify that the app is running.
For more information, see the [Pushing an App](https://docs.cloudfoundry.org/devguide/deploy-apps/deploy-app.html) topic.

### Test a deployed app
Use the cf CLI to review information and administer your app and your account. For example, you could edit the `manifest.yml` file to increase the number of app instances from 1 to 3 or redeploy the app with a new app name.

### Manage your app with the cf CLI
Run `cf help` to view a complete list of commands and run `cf help COMMAND` for detailed information about a specific command. For more information about using the cf CLI, refer to the cf CLI topics, especially the [Getting Started with the cf CLI](http://docs.cloudfoundry.org/cf-cli/getting-started.html) topic.

### Troubleshooting
If your app fails to start, verify that the app starts in your local environment. Refer to the [Troubleshooting Application Deployment and Health](https://docs.cloudfoundry.org/devguide/deploy-apps/troubleshoot-app-health.html) topic to learn more about troubleshooting.