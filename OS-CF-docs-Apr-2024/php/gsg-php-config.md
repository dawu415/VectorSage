# Configuring the PHP buildpack
You can modify the configuration file for your PHP buildpack.
The PHP buildpack stores all default configuration settings in the [defaults](https://github.com/cloudfoundry/php-buildpack/tree/master/defaults) directory of the Cloud Foundry PHP Buildpack repository on GitHub.

## options.json file
The `options.json` file is the configuration file for the buildpack itself. It instructs the buildpack what to download, where to download it from, and how to install it. It allows you to configure:

* Package names and versions, such as PHP, HTTPD, or Nginx versions.

* The web server to use, such as HTTPD, Nginx, or no server.

* The PHP extensions that are enabled.
The buildpack overrides the default `options.json` file with any configuration it finds in the `.bp-config/options.json` file of your app.
Here are explanations of the common options you might need to change:
| Variable | Explanation |
| --- | --- |
| WEB\_SERVER | Sets the web server to use. Must be one of `httpd`, `nginx`, or `none`. This value defaults to `httpd`. |
| HTTPD\_VERSION | Sets the version of Apache HTTPD to use. Currently the build pack supports the latest stable version. This value defaults to the latest release that is supported by the build pack. |
| ADMIN\_EMAIL | The value used in HTTPD’s configuration for [ServerAdmin](http://httpd.apache.org/docs/2.4/mod/core.html#serveradmin) |
| NGINX\_VERSION | Sets the version of Nginx to use. By default, the buildpack uses the latest stable version. |
| PHP\_VERSION | Sets the version of PHP to use. Set to a minor instead of a patch version, such as `"{PHP_70_LATEST}"`. See [options.json](https://github.com/cloudfoundry/php-buildpack/blob/4f5e50fabf66c5840210cbc64fcd4068d8a27448/cf_spec/fixtures/php_app_using_php_7_latest/.bp-config/options.json). |
| PHP\_EXTENSIONS | (DEPRECATED) A list of the [extensions](https://docs.cloudfoundry.org/buildpacks/php/gsg-php-config.html#php-extensions) to enable. `bz2`, `zlib`, `curl`, and `mcrypt` are enabled by default. |
| ZEND\_EXTENSIONS | A list of the Zend extensions to enable. Nothing is enabled by default. |
| APP\_START\_CMD | When the `WEB_SERVER` option is set to `none`, this command is used to start your app. If `WEB_SERVER` and `APP_START_CMD` are not set, then the buildpack searches, in order, for `app.php`, `main.php`, `run.php`, or `start.php`. This option accepts arguments. |
| WEBDIR | The root directory of the files served by the web server specified in `WEB_SERVER`. Defaults to `htdocs`. Other common settings are `public`, `static`, or `html`. The path is relative to the root of your app. |
| LIBDIR | This path is added to PHP’s `include_path`. Defaults to `lib`. The path is relative to the root of your app. |
| HTTP\_PROXY | The buildpack downloads uncached dependencies using HTTP. If you are using a proxy for HTTP access, set its URL here. |
| HTTPS\_PROXY | The buildpack downloads uncached dependencies using HTTPS. If you are using a proxy for HTTPS access, set its URL here. |
| ADDITIONAL\_PREPROCESS\_CMDS | A list of additional commands that run prior to the app starting. For example, you might use this command to run migration scripts or static caching tools before the app launches. |
For details about supported versions, see the release notes for your buildpack version on the [Releases](https://github.com/cloudfoundry/php-buildpack/releases) page of the Cloud Foundry PHP Buildpack repository on GitHub.

### HTTPD, Nginx, and PHP configuration
The buildpack automatically configures HTTPD, Nginx, and PHP for your app. This section explains how to modify the configuration.
The `.bp-config` directory in your app can contain configuration overrides for these components. Name the directories `httpd`, `nginx`, and `php`. Cloud Foundry recommends that you use [php.ini.d](https://docs.cloudfoundry.org/buildpacks/php/gsg-php-config.html#php_ini_d) or [fpm.d](https://docs.cloudfoundry.org/buildpacks/php/gsg-php-config.html#fpm_d).
If you override the `php.ini` or `php-fpm.conf` files, many other forms of configuration do not work.
For example:
```
.bp-config
httpd
nginx
php
```
Each directory can contain configuration files that the component understands.
For example, to change HTTPD logging configuration, run:
```
ls -l .bp-config/httpd/extra/
total 8

-rw-r--r-- 1 daniel staff 396 Jan 3 08:31 httpd-logging.conf
```
In this example, the `httpd-logging.conf` file overrides the one provided by the buildpack. Cloud Foundry recommends that you copy the default from the buildpack and modify it.
You can find the default configuration files in the [PHP Buildpack `/defaults/config` directory](https://github.com/cloudfoundry/php-buildpack/tree/master/defaults/config).
You must be careful when modifying configurations, as doing so can cause your app to fail, or cause Cloud Foundry to fail to stage your app.
You can add your own configuration files. The components do not reference them, so you must ensure that they are included. For example, you can add an include directive to the [httpd configuration](https://github.com/cloudfoundry/php-buildpack/blob/master/defaults/config/httpd/httpd.conf) to include your file:
```
ServerRoot "${HOME}/httpd"
Listen ${PORT}
ServerAdmin "${HTTPD_SERVER_ADMIN}"
ServerName "0.0.0.0"
DocumentRoot "${HOME}/#{WEBDIR}"
Include conf/extra/httpd-modules.conf
Include conf/extra/httpd-directories.conf
Include conf/extra/httpd-mime.conf
Include conf/extra/httpd-logging.conf
Include conf/extra/httpd-mpm.conf
Include conf/extra/httpd-default.conf
Include conf/extra/httpd-remoteip.conf
Include conf/extra/httpd-php.conf
Include conf/extra/httpd-my-special-config.conf # This line includes your additional file.
```

#### .bp-config/php/php.ini.d/
The buildpack adds any `.bp-config/php/php.ini.d/FILE-NAME.ini` files it finds in the app to the PHP configuration.
You can use this to change any value acceptable to `php.ini`. For a list of directives, see <http://php.net/manual/en/ini.list.php>.
For example, adding a file `.bp-config/php/php.ini.d/something.ini` to your app, with the following contents, overrides both the default charset and mimetype:
```
default_charset="UTF-8"
default_mimetype="text/xhtml"
```

##### Precedence
In order of highest precedence, PHP configuration values come from the following sources:

* PHP scripts using `ini_set()` to manually override config files

* `user.ini` files for local values

* `.bp-config/php/php.ini.d` to override main value, but not local values from user.ini files

#### .bp-config/php/fpm.d/
The buildpack adds any files it finds in the app under `.bp-config/php/fpm.d` that end with `.conf` (i.e `my-config.conf`) to the PHP-FPM configuration. You can use this to change any value acceptable to `php-fpm.conf`. For a list of directives, see <http://php.net/manual/en/install.fpm.configuration.php>.
PHP FPM config snippets are included by the buildpack into the global section of the configuration file. If you need to apply configuration settings for a PHP FPM worker, you must indicate this in your configuration file.
For example:
```
; This option is specific to the `www` pool
[www]
catch_workers_output = yes
```

### PHP Extensions
The buildpack adds any `.bp-config/php/php.ini.d/FILE-NAME.ini` files it finds in the app to the PHP configuration. You can use this to enable PHP or ZEND extensions. For example:
```
extension=redis.so
extension=gd.so
zend_extension=opcache.so
```
If an extension is already present and enabled in the compiled PHP, such as `intl`, you do not need to explicitly enable it to use that extension.

#### PHP\_EXTENSIONS vs. ZEND\_EXTENSIONS
PHP has two kinds of extensions, *PHP extensions* and *Zend extensions*. These hook into the PHP executable in different ways. For more information about the way extensions work internally in the engine, see <https://wiki.php.net/internals/extensions>.
Because they hook into the PHP executable in different ways, they are specified differently in ini files. Apps fail if a Zend extension is specified as a PHP extension, or a PHP extension is specified as a Zend extension.
If you see the following error:
```
php-fpm | [warn-ioncube] The example Loader is a Zend-Engine extension and not a module (pid 40)
php-fpm | [warn-ioncube] Please specify the Loader using 'zend_extension' in php.ini (pid 40)
php-fpm | NOTICE: PHP message: PHP Fatal error: Unable to start example Loader module in Unknown on line 0
```
Then move the `example` extension from `extension` to `zend_extension` and re-push your app.
If you see the following error:
```
NOTICE: PHP message: PHP Warning: example MUST be loaded as a Zend extension in Unknown on line 0
```
Then move the `example` extension from `zend_extension` to `extension` and re-push your app.

## Buildpack extensions
The buildpack comes with extensions for its default behavior. These are the [HTTPD](https://github.com/cloudfoundry/php-buildpack/tree/master/lib/httpd), [Nginx](https://github.com/cloudfoundry/php-buildpack/tree/master/lib/nginx), [PHP](https://github.com/cloudfoundry/php-buildpack/tree/master/lib/php), and [NewRelic](https://github.com/cloudfoundry/php-buildpack/tree/master/extensions/newrelic) extensions.
The buildpack is designed with an extension mechanism, allowing app developers to add behavior to the buildpack without modifying the buildpack code.
When you push an app, the buildpack runs any extensions found in the `.extensions` directory of your app.
For more information about writing extension, see the [Cloud Foundry PHP Buildpack](https://github.com/cloudfoundry/php-buildpack/blob/master/README.md) repository on GitHub.