# Windows Gemfile support
Learn how you can handle Ruby buildpack dependencies on Microsoft Windows machines.

## Windows Gemfiles
When a `Gemfile.lock` is generated on a Windows machine, it often contains gems
with Windows specific versions. This results in versions of gems such as
`mysql2`, `thin`, and `pg` containing “-x86-mingw32.” For example, the `Gemfile` might contain the following:
```
gem 'sinatra'
gem 'mysql2'
gem 'json'
```
When you run `bundle install` with `Gemfile` on a Windows machine, it results in the following `Gemfile.lock`:
```
GEM remote: http://rubygems.org/
specs:
json (1.7.3)
mysql2 (0.3.11-x86-mingw32)
rack (1.4.1)
rack-protection (1.2.0)
rack sinatra (1.3.2)
rack (~> 1.3, >= 1.3.6)
rack-protection (~> 1.2)
tilt (~> 1.3, >= 1.3.3)
tilt (1.3.3)
PLATFORMS x86-mingw32
DEPENDENCIES
json
mysql2
sinatra
```
Notice the “-x86-mingw32” in the version number of `mysql2`. Since Cloud Foundry runs on Linux machines, this fails to install. To mitigate this,
the Ruby Buildpack removes the `Gemfile.lock` and uses Bundler to resolve dependencies
from the `Gemfile`.

**Important**
Removing the `Gemfile.lock` causes dependency versions to be resolved
from the `Gemfile`. This could result in different versions being installed than
those listed in the `Gemfile.lock`.