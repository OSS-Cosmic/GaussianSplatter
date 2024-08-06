```
cat $(buck2 bxl tf//build_defs/compile_command.bxl:gen_compile_command -- --platform tf_platform//:linux_x86_64 --filter tf//... --filter root//:GaussianSplatter) > compile_commands.json
```

