# App session data
When your application has one instance, it’s generally safe to use the default session storage, which is the local file system.
You can only see problems if your single instance crashes, the local file system goes away, and you lose your sessions.
For many applications, this works just fine, but you need to consider how this affects your application.
If you have multiple application instances or you need a more robust solution for your application, use
Redis or Memcached as a backup store for your session data. The buildpack supports both backups and when one is bound to your application, it detects it and configures PHP to use it for session storage.
By default, there is no configuration necessary. To create a Redis or Memcached service, ensure that the service name contains `redis-sessions` or `memcached-sessions` and then bind the service to the application.
Example:
```
$ cf create-service redis some-plan app-redis-sessions
$ cf bind-service app app-redis-sessions
$ cf restage app
```
If you want to use a specific service instance or change the search key, you can set either `REDIS_SESSION_STORE_SERVICE_NAME` or `MEMCACHED_SESSION_STORE_SERVICE_NAME` in `.bp-config/options.json` to the new search key. The session configuration extension searches the bound services by name for the new session key.

## Configuration changes
When detected, the following changes are made:

### Redis

* the `redis` PHP extension is installed, which provides the session save handler

* `session.name` is set to `PHPSESSIONID` which deactivates sticky sessions

* `session.save_handler` is configured to `redis`

* `session.save_path` is configured based on the bound credentials, for example `tcp://host:port?auth=pass`

### Memcached

* the `memcached` PHP extension is installed, which provides the session save handler

* `session.name` is set to `PHPSESSIONID` which deactivates sticky sessions

* `session.save_handler` is configured to `memcached`

* `session.save_path` is configured based on the bound credentials (i.e. `PERSISTENT=app_sessions host:port`)

* `memcached.sess_binary` is set to `On`

* `memcached.use_sasl` is set to `On`, which enables authentication

* `memcached.sess_sasl_username` and `memcached.sess_sasl_password` are set with the service credentials