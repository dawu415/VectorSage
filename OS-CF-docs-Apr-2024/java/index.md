# Java buildpack
You can use the Java buildpack with apps written in Grails, Play, Spring, or any other JVM-based language or framework.
For information about specific versions, see [Java Buildpack Release Notes](https://github.com/cloudfoundry/java-buildpack/releases).
You can find the source for the Java buildpack in the [Java buildpack repository](https://github.com/cloudfoundry/java-buildpack) on GitHub.

## Buildpack and application logging
The Java buildpack only runs during the staging process, and only logs
staging information such as the downloaded components, configuration data, and work performed on your application by the buildpack.
The Java buildpack source documentation states the following:

* The Java buildpack logs all messages, regardless of severity, to
`APP-DIRECTORY/.java-buildpack.log`. The buildpack also logs messages to `$stderr`, filtered by a configured severity level.

* If the buildpack fails with an exception, the exception message is logged with
a log level of `ERROR`. The exception stack trace is logged with a log
level of `DEBUG`. This prevents users from seeing stack traces by default.
Once staging completes, the buildpack stops logging then the Loggregator handles application logging.
Your application must write to STDOUT or STDERR for its logs to be included in
the Loggregator stream.
For more information about logging, see [App Logging in Cloud Foundry](https://docs.cloudfoundry.org/devguide/deploy-apps/streaming-logs.html).

## BOSH custom trusted certificate support
Versions 3.7 and later of the Java buildpack support BOSH configured custom trusted certificates.
For more information, see [Configuring Trusted Certificates](http://bosh.io/docs/trusted-certs.html) in the BOSH documentation.
The Java buildpack pulls the contents of `/etc/ssl/certs/ca-certificates.crt` and `$CF_INSTANCE_CERT/$CF_INSTANCE_KEY` by default.
The log output for Diego Instance Identity-based `KeyStore` appears as follows:
```
Adding System Key Manager
Adding Key Manager for /etc/cf-instance-credentials/instance.key and /etc/cf-instance-credentials/instance.crt
Start watching /etc/cf-instance-credentials/instance.crt
Start watching /etc/cf-instance-credentials/instance.key
Initialized KeyManager for /etc/cf-instance-credentials/instance.key and /etc/cf-instance-credentials/instance.crt
```
The log output for Diego Trusted Certificate-based `TrustStore` appears as follows:
```
Adding System Trust Manager
Adding TrustManager for /etc/ssl/certs/ca-certificates.crt
Start watching /etc/ssl/certs/ca-certificates.crt
Initialized TrustManager for /etc/ssl/certs/ca-certificates.crt
```

## Memory constraints
The memory calculator in the Java buildpack accounts for the following memory regions:

* `-Xmx`: Heap

* `-XX:MaxMetaspaceSize`: Metaspace

* `-Xss`: Thread Stacks

* `-XX:MaxDirectMemorySize`: Direct Memory

* `-XX:ReservedCodeCacheSize`: Code Cache

* `-XX:CompressedClassSpaceSize`: Compressed Class Space
Most applications run if they use the Cloud Foundry default container size of 1 G without any modifications.
However, you can configure those memory regions directly as needed.
The Java buildpack optimizes for all non-heap memory regions first and leaves the remainder for the heap.
The Java buildpack prints a histogram of the heap to the logs when the JVM encounters a terminal failure.