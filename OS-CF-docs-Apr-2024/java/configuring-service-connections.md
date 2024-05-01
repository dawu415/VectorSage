# Configuring service connections
The path for configuring service access in your Java based applications is the [java-cfenv](https://github.com/pivotal-cf/java-cfenv) library.
This library can read and parse `VCAP_SERVICES` and help you extract the information for use in your application.
There are a number of ways to implement this and all Java applications can use the library; it is not limited to specific frameworks.
To get started, you must first add a dependency in your project for the library.

## Dependencies
The following examples show the dependency syntax for Maven and Gradle.
For Maven:
```
<dependency>
<groupId>io.pivotal.cfenv</groupId>
<artifactId>java-cfenv</artifactId>
<version>2.4.0</version>
</dependency>
```
For Gradle:
```
implementation "io.pivotal.cfenv:java-cfenv:2.4.0"
```

## Java-only / No framework
The entry point for the library is the class `CfEnv`, which parses Cloud Foundry environment variables. For example, `VCAP_SERVICES`. `VCAP_SERVICES` which contains a JSON string that includes credential information used to access bound services, for example, databases.
Create a `CfEnv` instance and use its `findCredentialsBy*` methods. There are methods for finding by label, name, and tag. Multiple strings can be passed to match against more than one tag, and the finder method supports passing a regex string for pattern matching.
For example:
```
CfEnv cfEnv = new CfEnv();
String redisHost = cfEnv.findCredentialsByTag("redis").getHost();
String redisPort = cfEnv.findCredentialsByTag("redis").getPort();
String redisPassword = cfEnv.findCredentialsByTag("redis").getPassword();
List<CfService> cfService = cfEnv.findAllServices();
CfService redisService = cfEnv.findServiceByTag("redis");
List<String> redisServiceTags = redisService.getTags();
String redisPlan = redisService.getPlan();
redisPlan = redisService.get("plan")
CfCredentials redisCredentials = cfEnv.findCredentialsByTag("redis");
String redisPort = redisCredentials.getPort();
Integer redisPort = redisCredentials.getMap().get("port");
cfService = cfEnv.findServiceByName("redis");
cfService = cfEnv.findServiceByLabel("p-redis");
cfService = cfEnv.findServiceByLabel(".\*-redis");
```

### JDBC support
There is additional support for getting a JDBC URL from a service binding. This support is contained in the module `java-cfenv-jdbc`. To enable this module, add the appropriate dependency.
For Maven:
```
<dependency>
<groupId>io.pivotal.cfenv</groupId>
<artifactId>java-cfenv-jdbc</artifactId>
<version>2.4.0</version>
</dependency>
```
For Gradle:
```
implementation "io.pivotal.cfenv:java-cfenv-jdbc:2.4.0"
```
The entry point for this feature is the class `CfJdbcEnv`, which is a subclass of `CfEnv` and adds a few methods. The method `findJdbcService` heuristically looks at all services for known tags, labels, and names of common database services to create the URL.
For example:
```
CfJdbcEnv cfJdbcEnv = new CfJdbcEnv()
CfJdbcService cfJdbcService = cfJdbcEnv.findJdbcService();
String jdbcUrl = cfJdbcService.getJdbcUrl();
String username = cfJdbcService.getUsername();
String password = cfJdbcService.getPassword();
String driverClassName = cfJdbcService.getDriverClassName();
```

## Spring Framework
The Spring Framework provides additional support for application developers.

### Spring Expression Language
If you register the `CfJdbcEnv` class as a bean, then you can use the Spring Expression Language to set properties.
```
@Bean
public CfEnv cfEnv() {
return new CfEnv();
}
```
Then, in the properties file imported by Spring, refer to the `CfEnv` bean using the following syntax:
```
cassandra.contact-points=#{ cfEnv.findCredentialsByTag('cassandra').get('node_ips') }
cassandra.username=#{ cfEnv.findCredentialsByTag('cassandra').getUserName() }
cassandra.password=#{ cfEnv.findCredentialsByTag('cassandra').getPassword() }
cassandra.port=#{ cfEnv.findCredentialsByTag('cassandra').get('cqlsh_port') }
```
To specifically target JDBC databases, register this instead:
```
@Bean
public CfJdbcEnv cfJdbcEnv() {
return new CfJdbcEnv();
}
```
Then in a property file imported by Spring, refer to the `CfJdbcEnv` bean using the following syntax:
```
myDatasourceUrl=#{ cfJdbcEnv.findJdbcService().getUrl() }
```

### Spring Boot
The module `java-cfenv-boot` provides several `EnvironmentPostProcessor` implementations that set well-known Spring Boot properties so that Spring Boot’s auto-configuration is active. For example, the `CfDataSourceEnvironmentPostProcessor` sets the Spring Boot property, `spring.datasource.url`.
To use these, add a dependency on `java-cfenv-boot`.
For Maven:
```
<dependency>
<groupId>io.pivotal.cfenv</groupId>
<artifactId>java-cfenv-boot</artifactId>
<version>2.4.0</version>
</dependency>
```
For Gradle:
```
implementation "io.pivotal.cfenv:java-cfenv-boot:2.4.0"
```
The list of supported services are:

* Databases - DB2, MySQL, Oracle, PostgreSQL, SQL Server

* RabbitMQ

* Cassandra

* MongoDB

* Redis

* CredHub

* HashiCorp Vault
If you need to prevent processing of a specific service instance, set the flag in your application properties to:
```
cfenv.service.{serviceName}.enabled=false
```

## Migrating from Spring AutoReconfiguration and Spring Cloud Connectors
The `java-cfenv` library replaces the older Spring AutoReconfiguration and Spring Cloud Connectors libraries. Use the information in the following sections to migrate to `java-cfenv`.

### Change dependencies
Remove references to any of these libraries from the application build files.
```
org.springframework.boot:spring-boot-starter-cloud-connectors
```
or
```
org.springframework.cloud:spring-cloud-core
org.springframework.cloud:spring-cloud-connectors-core
org.springframework.cloud:spring-cloud-cloudfoundry-connector
org.springframework.cloud:spring-cloud-spring-service-connector
```
Then add a reference to the [`java-cfenv` library](https://docs.cloudfoundry.org/buildpacks/java/configuring-service-connections.html#dependencies).

### Code changes
Remove any of the `@ServiceScan` or `@CloudScan` annotations from Spring Java configuration classes (provided by Spring Cloud Connectors). Replace them with the [Spring SPeL or Spring Boot configuration options](https://docs.cloudfoundry.org/buildpacks/java/configuring-service-connections.html#spring) listed above.

### Migration considerations
Review these additional considerations before you migrate.

* **Non-Spring Boot applications:**
If you have a Spring Application that is a non-Spring Boot application, you can still migrate to `java-cfenv`. You must use either the [no framework options](https://docs.cloudfoundry.org/buildpacks/java/configuring-service-connections.html#no-framework) or the [Spring SPeL option](https://docs.cloudfoundry.org/buildpacks/java/configuring-service-connections.html#spring-spel). With SPeL, you might need to manually process the expressions, depending on where you are configuring them. See the Spring documentation for places where SPeL expressions are processed by default.

* **Multiple service instances:**
Spring Cloud Connectors support connections to multiple service instances.
If you need to configure connections to multiple instances of a given service type, or do anything more than setting application properties for Spring Boot to pick up and use in auto-configuration, then you must follow the manual configuration approaches laid out in the sections above to access the binding credentials. Either [with direct Java code](https://docs.cloudfoundry.org/buildpacks/java/configuring-service-connections.html#no-framework) or with [SPeL](https://docs.cloudfoundry.org/buildpacks/java/configuring-service-connections.html#spring-spel). Then follow the same procedure that is used to connect to the services in any other non-Cloud Foundry deployment environment.

* **Code modifications:**
The Java Buildpack injects the Spring Auto Reconfiguration module code into your application and overwrites your service configuration. This works well in some cases, but sometimes it causes problems.
With `java-cfenv`, there is no auto-reconfiguration magic. You can explicitly configure your services or you can use the [Spring Boot mappers](https://docs.cloudfoundry.org/buildpacks/java/configuring-service-connections.html#spring-boot). The [Spring Boot mappers](https://docs.cloudfoundry.org/buildpacks/java/configuring-service-connections.html#spring-boot) are the option most similar to previous operation. Note, that when things don’t work, it’s generally clearer what happened, and it’s easier to debug the problem.

* **Cloud property placeholders:**
The Spring Auto Reconfiguration module exposes a set of property placeholder values that you can use to access values from `VCAP_SERVICES`. If you are using these placeholders, then you must switch from using `cloud.<property>`. Use `vcap.<property>` instead.
Spring Boot exposes the same information, just under the `vcap.` prefix instead of the `cloud.` prefix.

* **Spring Cloud Profile:**
The Spring Auto Reconfiguration module enables a Spring Profile called `cloud`, by default. Users have come to expect this behavior when deploying to Cloud Foundry. Without the Spring Auto Reconfiguration module, you do not get this behavior. Fortunately, you can enable it using one of these methods:

+ Run `cf set-env <APP> SPRING_PROFILES_ACTIVE cloud`

+ Add `SPRING_PROFILES_ACTIVE: cloud` to the `env:` block in your `manifest.yml` file. This supplies the list of profiles for Spring to use.If you need to set additional profiles, you can use `SPRING_PROFILES_INCLUDE` instead. This appends to the existing set of profiles.

* **Spring Cloud Connector extensions:**
If you have created any custom Spring Cloud Connector extensions, you must migrate them to `java-cfenv`. This requires two steps:

1. Write a [Spring Boot Auto Configuration](https://docs.spring.io/spring-boot/docs/current/reference/html/boot-features-developing-auto-configuration.html) library that creates connections to your service from Spring configuration properties. This makes it easy to also use it in a non-cloud Spring Boot app. When this is done correctly, you can to use the library, and set properties in `application.properties` (or through other means), and you can have a connection to your service.

2. [Write a java-cfenv extension](https://github.com/pivotal-cf/java-cfenv#supporting-other-services). This takes values from `VCAP_SERVICES` and maps them to the properties that you exposed with your Spring Boot Auto Configuration library from the previous step.

### Java Buildpack warnings
The Java Buildpack generates warnings to help with migrating from Spring Cloud Connectors and Spring Auto Reconfiguration.

* **Spring Auto Reconfiguration installed:**
This is the message that is generated when the buildpack installs the Spring Auto Reconfiguration JAR. This happens by default, and notifies you that it is happening.
```
[SpringAutoReconfiguration] WARN ATTENTION: The Spring Auto Reconfiguration and shaded Spring Cloud Connectors libraries are being installed. These projects have been deprecated, are no longer receiving updates, and should not be used going forward.
[SpringAutoReconfiguration] WARN If you are not using these libraries, set `JBP_CONFIG_SPRING_AUTO_RECONFIGURATION='{enabled: false}'` to their installation and clear this warning message. The buildpack switches the default to deactivate by default after Aug 2022. Spring Auto Reconfiguration and its shaded Spring Cloud Connectors are removed from the buildpack after Dec 2022.
[SpringAutoReconfiguration] WARN If you are using these libraries, please migrate to java-cfenv immediately. See https://via.vmw.com/EhzD for migration instructions
```
How you resolve this depends on whether you are depending on the Auto Reconfiguration to occur.

+ If your application depends on Auto Reconfiguration behavior, then you must make code changes in your application to use the `java-cfenv` library. See the instructions above for how to include the dependencies and how to access service information using this library.
After you have added `java-cfenv` to your classpath, the Java buildpack no longer installs the Auto Reconfiguration JAR and you no longer see this message.

+ If your application is not depending on Auto Reconfiguration behavior, and has already been updated to `java-cfenv`, then this message is not generated. If your application does not use Auto Reconfiguration or `java-cfenv`, then you can either:

+ Run `cf set-env <APP> JBP_CONFIG_SPRING_AUTO_RECONFIGURATION '{enabled: false}'`

+ Add `JBP_CONFIG_SPRING_AUTO_RECONFIGURATION: '{enabled: false}'` to the `env:` block in your `manifest.yml` file.
Alternatively, you can use a buildpack released after Aug 2022, after which this feature is no longer enabled by default.

* **Spring Cloud Connectors present:**
The following message is generated when the buildpack detects that the Spring Cloud Connectors library is present on the classpath.
```
[SpringAutoReconfiguration] WARN ATTENTION: The Spring Cloud Connectors library is present in your application. This library has been in maintenance mode since July 2019 and stops receiving all updates after Dec 2022.
[SpringAutoReconfiguration] WARN Please migrate to java-cfenv immediately. See <https://via.vmw.com/EhzD> for migration instructions.
```
When this message appears, it means that your application or one of its dependencies is including the Spring Cloud Connectors library. You must remove it and migrate to `java-cfenv`.
After you have migrated to `java-cfenv`, and the Spring Cloud Connectors libraries are no longer on your classpath, this error no longer appears. There is no way to manually suppress this message.