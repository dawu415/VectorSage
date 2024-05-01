# Running tasks in your apps
A task is an app or script, the code of which is included as part of a deployed app, but runs
independently in its own container. This topic describes how to run tasks in Cloud Foundry.

## Tasks in Cloud Foundry
In contrast to a long-running process (LRP), tasks run for a finite amount of time, then stop. Tasks run in their own containers and are designed to use minimal resources. After a task runs, Cloud Foundry destroys the container running the task.
As a single use object, a task can be checked for its state and for a success or failure message.

### Use cases for tasks
Tasks are used to perform one-off jobs, which include:

* Migrating a database

* Sending an email

* Running a batch job

* Running a data processing script

* Processing images

* Optimizing a search index

* Uploading data

* Backing-up data

* Downloading content

### How tasks are run
Tasks are always run asynchronously, meaning that they run independently from the parent app or other tasks that run on the same app.
The life cycle of a task is as follows:

1. A user initiates a task in Cloud Foundry using one of the following mechanisms:

* The `cf run-task APP-NAME "TASK"` command. For more information, see [Running tasks in your apps](https://docs.cloudfoundry.org/devguide/using-tasks.html#run-tasks).

* A Cloud Controller v3 API call. For more information, see the [Cloud Foundry API documentation](http://v3-apidocs.cloudfoundry.org/version/3.0.0/index.html#tasks).

* The Cloud Foundry Java Client. For more information, see [Cloud Foundry Java Client Library](https://docs.cloudfoundry.org/buildpacks/java/java-client.html) and the [Cloud Foundry Java Client](https://github.com/cloudfoundry/cf-java-client) repository on GitHub.

2. Cloud Foundry creates a container specifically for the task.

3. Cloud Foundry runs the task on the container using the value passed to the `cf run-task` command.

4. Cloud Foundry destroys the container.
The container also inherits environment variables, service bindings, and security groups bound to the app.

**Note**
You cannot SSH into the container running a task.

### Task logging and execution history
Any data or messages the task outputs to stdout or stderr is available in the firehose logs of the app. A syslog drain attached to the app receives the task log output.
The task execution history is retained for one month.

## Manage tasks
At the system level, a user with admin-level privileges can use the Cloud Controller v3 API to view all tasks that are running within an org or space. For
more information, see the [Cloud Foundry API documentation](http://v3-apidocs.cloudfoundry.org/version/3.0.0/index.html#list-tasks).
Admins can set the default memory, disk usage and log rate quotas for tasks on a global level.
Tasks use the same memory, disk usage, and log rate limit defaults as apps, unless you customize them using the `cf run-task` command. For more information
about the `cf run-task` command, see the [Cloud Foundry CLI reference guide](https://cli.cloudfoundry.org/en-US/v8/run-task.html).
In the BOSH Cloud Controller Worker job, the `cc.default_app_memory` and `cc.default_app_disk_in_mb` properties specify the default memory and disk
allocations for tasks.
The `cc.default_app_log_rate_limit_in_bytes_per_second` property specifies the default log rate limit. For more information about BOSH Cloud Controller Worker jobs, see
[BOSH documentation](https://bosh.io/jobs/cloud_controller_worker?source=github.com/cloudfoundry/capi-release).

## Run a task on an app
You can use the Cloud Foundry Command Line Interface (cf CLI) to run a task in the context of an app.

**Important**

* To run tasks with the cf CLI, you must install cf CLI v6.23.0 or later, or install cf CLI v7 or v8. To download, install, and uninstall the cf CLI, see [Installing the Cloud Foundry Command Line Interface](https://docs.cloudfoundry.org/cf-cli/install-go-cli.html).

* To run a task using cf CLI v6 without starting the app, push the app with `cf push -i 0` and then run the task. You can run the app later by scaling up its instance count.

### Run a task on an app with cf CLI v6
To run a task on an app with cf CLI v6:

1. In a terminal window, push your app by running:
```
cf push APP-NAME
```
Where `APP-NAME` is the name of your app,

2. Run your task on the deployed app by running:
```
cf run-task APP-NAME "TASK" --name TASK-NAME
```
Where:

* `APP-NAME` is the name of your app.

* `TASK` is the task you want to run.

* `TASK-NAME` is the name you want to give the task.
The following example command runs a database migration as a task on the `example-app` app:
```
cf run-task example-app "bin/rails db:migrate" --name example-task
```
When the task runs successfully, you see an output similar to the following example:
```
Creating task for app example-app in org example-org / space development as [admin@example.org](mailto:admin@example.org)...
OK
Task 1 has been submitted successfully for execution.
```

**Note**
To run a task again, you must run it as a new task using the previous command.

3. To display the recent logs of the app and all its tasks, run:
```
cf logs APP-NAME --recent
```
Where `APP-NAME` is the name of your app.
If a task succeeds, you see logs similar to the following example:
```
2017-01-03T15:58:06.57-0800 [APP/TASK/my-task/0]OUT Creating container
2017-01-03T15:58:08.45-0800 [APP/TASK/my-task/0]OUT Successfully created container
2017-01-03T15:58:13.32-0800 [APP/TASK/my-task/0]OUT D, [2017-01-03T23:58:13.322258 #7] DEBUG -- : (15.9ms) CREATE TABLE "schema_migrations" ("version" character varying PRIMARY KEY)
2017-01-03T15:58:13.33-0800 [APP/TASK/my-task/0]OUT D, [2017-01-03T23:58:13.337723 #7] DEBUG -- : (11.9ms) CREATE TABLE "ar_internal_metadata" ("key" character varying PRIMARY KEY, "value" character varying, "created_at" timestamp NOT NULL, "updated_at" timestamp NOT NULL)
2017-01-03T15:58:13.34-0800 [APP/TASK/my-task/0]OUT D, [2017-01-03T23:58:13.340234 #7] DEBUG -- : (1.6ms) SELECT pg_try_advisory_lock(3720865444824511725);
2017-01-03T15:58:13.35-0800 [APP/TASK/my-task/0]OUT D, [2017-01-03T23:58:13.351853 #7] DEBUG -- : ActiveRecord::SchemaMigration Load (0.7ms) SELECT "schema_migrations".* FROM "schema_migrations"
2017-01-03T15:58:13.35-0800 [APP/TASK/my-task/0]OUT I, [2017-01-03T23:58:13.357294 #7] INFO -- : Migrating to Createtopics (20161118225627)
2017-01-03T15:58:13.35-0800 [APP/TASK/my-task/0]OUT D, [2017-01-03T23:58:13.359565 #7] DEBUG -- : (0.5ms) BEGIN
2017-01-03T15:58:13.35-0800 [APP/TASK/my-task/0]OUT == 20161118225627 Createtopics: migrating ===================================
2017-01-03T15:58:13.50-0800 [APP/TASK/my-task/0]OUT Exit status 0
2017-01-03T15:58:13.56-0800 [APP/TASK/my-task/0]OUT Destroying container
2017-01-03T15:58:15.65-0800 [APP/TASK/my-task/0]OUT Successfully destroyed container
```
If a task fails, you see logs similar to the following example:
```
2016-12-14T11:09:26.09-0800 [APP/TASK/my-task/0]OUT Creating container
2016-12-14T11:09:28.43-0800 [APP/TASK/my-task/0]OUT Successfully created container
2016-12-14T11:09:28.85-0800 [APP/TASK/my-task/0]ERR bash: bin/rails: command not found
2016-12-14T11:09:28.85-0800 [APP/TASK/my-task/0]OUT Exit status 127
2016-12-14T11:09:28.89-0800 [APP/TASK/my-task/0]OUT Destroying container
2016-12-14T11:09:30.50-0800 [APP/TASK/my-task/0]OUT Successfully destroyed container
```
If your task name is unique, you can `grep` the output of the `cf logs` command for the task name to view task-specific logs.

### Run a task on an app with cf CLI v7
To run a task on an app with cf CLI v7:

1. Configure your v3 API manifest with a task as a process type. For more information, see the [Cloud Foundry API documentation](https://v3-apidocs.cloudfoundry.org/version/3.78.0/index.html#the-app-manifest-specification).

2. In a terminal, push your app by running:
```
cf push APP-NAME --task
```
Where `APP-NAME` is the name of your app.

3. Run your task on the deployed app by running:
```
cf run-task APP-NAME --name TASK-NAME
```
Where:

* `APP-NAME` is the name of your app.

* `TASK-NAME` is the name you want to give the task.

**Important**
`cf run-task` allows you to include the `--process` and `--command` flags. Including the `--command` flag overrides the manifest property.
The following example command runs a task on the `example-app` app:
```
cf run-task example-app --name example-task
```
When the task runs successfully, you see terminal output similar to the following example:
```
Creating task for app example-app in org example-org / space development as [admin@example.org](mailto:admin@example.org)...
OK
Task 1 has been submitted successfully for execution.
```

4. To display the recent logs of the app and all its tasks, run:
```
cf logs APP-NAME --recent
```
Where `APP-NAME` is the name of your app.

## List tasks running on an app
To list the tasks for a given app:

1. In a terminal window, run:
```
cf tasks APP-NAME
```
Where `APP-NAME` is the name of your app. The command returns output similar to the following example:
```
Getting tasks for app example-app in org example-org / space development as [admin@example.org](mailto:admin@example.org)...
OK
```
id name state start time command
2 339044ef FAILED Wed, 23 Nov 2016 21:52:52 UTC echo foo; sleep 100; echo bar
1 8d0618cf SUCCEEDED Wed, 23 Nov 2016 21:37:28 UTC bin/rails db:migrate
Tasks can be in these states:
| State | Description |
| --- | --- |
| `RUNNING` | The task is in progress. |
| `FAILED` | The task did not complete. This state occurs when a task does not work correctly or a user cancels the task. |
| `SUCCEEDED` | The task completed successfully. |

## Cancel a task
After you run a task, you might be able to cancel it before it finishes.
To cancel a running task:

1. In a terminal window, run:
```
cf terminate-task APP-NAME TASK-ID
```
Where:

* `APP-NAME` is the name of your app.

* `TASK-NAME` is the name you want to give the task.
The previous command returns output similar to the following example:
```
Terminating task 2 of app example-app in org example-org / space development as [admin@example.org](mailto:admin@example.org)...
OK
```