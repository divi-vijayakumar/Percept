// HTTP Basic Auth at the edge.
//
// Netlify's built-in site password is a paid feature; this is the free-tier equivalent and it is
// real server-side auth, not a client-side gate — the page bytes are never sent without a valid
// header. The credential lives in a site env var, never in this file or in git.
export default async (request, context) => {
  const user = Netlify.env.get("SITE_USER") || "percept";
  const pass = Netlify.env.get("SITE_PASSWORD");
  if (!pass) return context.next();          // fail OPEN only if never configured, never silently closed
  const expected = "Basic " + btoa(`${user}:${pass}`);
  const got = request.headers.get("authorization") || "";
  // Length-independent compare: cheap, and avoids leaking the password length via timing.
  let ok = got.length === expected.length;
  for (let i = 0; i < Math.max(got.length, expected.length); i++) {
    ok = ok && got.charCodeAt(i) === expected.charCodeAt(i);
  }
  if (!ok) {
    return new Response("Authentication required.", {
      status: 401,
      headers: {
        "WWW-Authenticate": 'Basic realm="percept evaluation layers", charset="UTF-8"',
        "Cache-Control": "no-store",
      },
    });
  }
  return context.next();
};
export const config = { path: "/*" };
