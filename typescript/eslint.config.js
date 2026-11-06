// @ts-check
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["scripts/**", "typecheck/**", "dist/**", "node_modules/**", "coverage/**"] },
  ...tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    rules: {
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
    },
  },
  {
    files: ["src/**/*.ts"],
    rules: {
      // The core imports nothing but the JS runtime and its own files (PRD 06 §13).
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            { group: ["node:*", "fs", "path", "fs/*"], message: "no Node built-ins in src/" },
            { regex: "^[^./]", message: "src/ imports only its own files (zero runtime dependencies)" },
          ],
        },
      ],
    },
  },
  {
    files: ["eslint.config.js", "tsup.config.ts", "vitest.config.ts"],
    extends: [tseslint.configs.disableTypeChecked],
  },
);
