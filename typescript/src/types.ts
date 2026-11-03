/** JSON-safe values: everything `toDict()` emits and `args` holds. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
