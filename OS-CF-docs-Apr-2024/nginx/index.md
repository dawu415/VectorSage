# NGINX buildpack
You can push your NGINX app to Cloud Foundry and configure your NGINX app to use the NGINX buildpack.

## Push an app
If your app contains an `nginx.conf` file, Cloud Foundry automatically uses the NGINX buildpack when you run `cf push` to deploy your app.
If your Cloud Foundry deployment does not have the NGINX buildpack installed or the installed version is outdated, deploy your app with the current buildpack by running:
```
cf push YOUR-APP -b https://github.com/cloudfoundry/nginx-buildpack.git
```
Where `YOUR-APP` is the name of your app.
For example:
```
$ cf push my-app -b https://github.com/cloudfoundry/nginx-buildpack.git
```

## Configure NGINX
Cloud Foundry recommends that you use the default NGINX directory structure for your NGINX web server. You can view this directory structure in the [nginx-buildpack](https://github.com/cloudfoundry/nginx-buildpack/tree/master/fixtures/mainline) repository in GitHub.
Configure the NGINX web server. You need these elements.

* A root directory for all static web content

* A MIME type configuration file

* An NGINX configuration file

* A `buildpack.yml` YAML file that defines the version of NGINX to use. For example, see [buildpack.yml](https://github.com/cloudfoundry/nginx-buildpack/blob/master/fixtures/mainline/buildpack.yml) Buildpack repository in GitHub.
Make any custom configuration changes based on these default files to verify compatibility with the buildpack.

## Create the nginx.conf file
Use the templating syntax when you create an `nginx.conf` file. This templating syntax loads modules and binds to ports
based on values known at start time.

### Port
Use `{{port}}` to set the port on which to listen. At start time, `{{port}}` interpolates in the value of `$PORT`.
You must use `{{port}}` in your `nginx.conf` file.
For example, to set an NGINX server to listen on `$PORT`, include the following in your `nginx.conf` file:
```
server {
listen {{port}};
}
```

### Name resolution
The NGINX buildpack does not resolve internal routes by default. To resolve internal routes, use `{{nameservers}}` to set the resolver IP address. At start time, `{{nameservers}}` interpolates the address of a platform-provided DNS service that includes information about internal routes.
Connections to internal routes do not go through the Cloud Foundry routing tier. As a result, you might see errors
if you proxy an app on an internal route while it is restarting. There are some workarounds you might need to consider.
For more information, see [Using DNS for Service Discovery with NGINX and NGINX Plus](https://www.nginx.com/blog/dns-service-discovery-nginx-plus/) on the NGINX blog.

### Environment variables
To use an environment variable, include `{{env "YOUR-VARIABLE"}}`, where `YOUR-VARIABLE` is the name of an environment variable. At staging and at startup, the current value of the environment variable is retrieved.
For example, include the following in your `nginx.conf` file to activate or deactivate GZipping based on the value of `GZIP_DOWNLOADS`:
```
gzip {{env "GZIP_DOWNLOADS"}};
```

* If you set `GZIP_DOWNLOADS` to `off`, NGINX does not GZip files.

* If you set `GZIP_DOWNLOADS` to `on`, NGINX GZips files.

### Unescaped environment variables
To use unescaped environment variables, add an array of environment variable names to the `buildpack.yml`. See the following example:
```

---
nginx:
version: stable
plaintext_env_vars:

- "OVERRIDE"
```
In this example, the `OVERRIDE` environment variable can contain `.json` content without being `html` escaped.
You must properly quote such variables to appear as strings in the `nginx.conf` file.

### Loading dynamic modules
NGINX can dynamically load modules at runtime. These modules are shared-object files that can be dynamically loaded using the `load_module` directive. In addition to loading modules dynamically, the NGINX version provided by the buildpack has statically compiled the following modules into the NGINX binary:

* `ngx_http_ssl_module`

* `ngx_http_realip_module`

* `ngx_http_gunzip_module`

* `ngx_http_gzip_static_module`

* `ngx_http_auth_request_module`

* `ngx_http_random_index_module`

* `ngx_http_secure_link_module`

* `ngx_http_stub_status_module`

* `ngx_http_sub_module`
These statically compiled modules do not need to be loaded at runtime and are already available for use.
To load a dynamic NGINX module, use the following syntax in the app `nginx.conf` file for your app:
```
{{module "MODULE-NAME"}}
```
If you have provided a module in a `modules` directory located at the root of your app, the buildpack instructs NGINX to load that module.
If you have not provided a module, the buildpack instructs NGINX to search for a matching built in dynamic module.
As of v0.0.5 of the buildpack, the `ngx_stream_module` is available as a dynamic module that is built into the buildpack.
For example, to load a custom module named `ngx_hello_module`, provide a `modules/ngx_hello_module.so` file in your app directory and add the line below to the top of your `nginx.conf` file:
```
{{module "ngx\_hello\_module"}}
```
To load a built in module like `ngx_stream_module`, add the following line to the top of your `nginx.conf` file. You do not need to provide an `ngx_stream_module.so` file:
```
{{module "ngx\_stream\_module"}}
```
To name your modules directory something other than `modules`, use the NGINX `load_module` directive, providing a path to the module relative to the location of your `nginx.conf` file. For example:
`load_module some_module_dir/my_module.so`

### Enable logging
By default, logging is deactivated in the NGINX buildpack. This helps optimize performance.
If you configure NGINX to log to stdout or stderr, the logs are captured by the Cloud Foundry logging subsystem.

#### Logging access
Use the `access_log` directive in the appropriate location in the `nginx.conf` to enable access logging. Use the following syntax:
```
access_log &lt;file&gt; [format]
```
Where:

* `file` is the name of the file where the log is to be stored. The special value `/dev/stdout` selects the standard output.

* `format` is the logging format to use. Refer to [the NGINX documentation for valid customization options](https://nginx.org/en/docs/http/ngx_http_log_module.html#log_format).
Example:
```
access_log /dev/stdout;
```

#### Logging errors
Set the debug level with the `error_log` directive in the `nginx.conf` to enable debug logging. Add the `error_log` entry to your `nginx.conf` file using the syntax below:
```
error_log &lt;file&gt; [level];
```
Where:

* `file` is the name of the file where the log is to be stored. The special value `stderr` selects the standard error file. Logging to `syslog` can be configured by specifying the “`syslog:`” prefix.

* `level` specifies the level of logging, and can be one of the following:

+ `debug`

+ `info`

+ `notice`

+ `warn`

+ `error` (default)

+ `crit`

+ `alert`

+ `emerg`
The log levels above are listed in the order of increasing severity. Setting a certain log level causes all messages of the specified and
more severe log levels to be logged. For example, the default level `error` causes error, crit, alert, and emerg messages to be
logged. If this parameter is omitted, then `error` is used.
Example:
```
error_log stderr debug;
```

#### Debug logs for selected clients
To enable the debugging log for selected client addresses only, use the syntax below.
```
error_log stderr;
events {
debug_connection 192.168.1.1;
debug_connection 192.168.10.0/24;
}
```

#### Logging to a cyclic memory buffer
The debugging log can be written to a cyclic memory buffer. Use the syntax shown below.
```
error_log memory:32m debug;
```
Logging to the memory buffer on the debug level does not have significant impact on performance, even under high load. In this case, the
log can be extracted using a gdb script like the one in the following example:
```
set $log = ngx_cycle->log
while $log->writer != ngx_log_memory_writer
set $log = $log->next
end
set $buf = (ngx_log_memory_buf_t *) $log->wdata
dump binary memory debug_log.txt $buf->start $buf->end
```

## NGINX buildpack support
The resources listed in this section can assist you when using the NGINX buildpack or when developing your own NGINX buildpack.

* **NGINX Buildpack Repository in GitHub**: Find more information about using and extending the NGINX buildpack in the [NGINX buildpack](https://github.com/cloudfoundry/nginx-buildpack) GitHub repository.

* **Release Notes**: Find current information about this buildpack on the [NGINX buildpack release page](https://github.com/cloudfoundry/nginx-buildpack/releases) in GitHub.

* **Slack**: Join the #buildpacks channel in the [Cloud Foundry Slack community](http://slack.cloudfoundry.org/).