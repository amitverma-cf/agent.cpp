// agent.qozu.fyi serves the two install scripts of the project and nothing else (no pages, no docs).
//   curl -fsSL https://agent.qozu.fyi | sh        (Linux)
//   irm https://agent.qozu.fyi | iex              (Windows PowerShell)
// The scripts are read from the repository's main branch, so changing them needs no redeploy.
const RAW = "https://raw.githubusercontent.com/amitverma-cf/agent.cpp/main/scripts/";

export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (request.method !== "GET" && request.method !== "HEAD") return new Response("method not allowed\n", { status: 405 });
    let file;
    if (url.pathname === "/install.sh") file = "install.sh";
    else if (url.pathname === "/install.ps1") file = "install.ps1";
    else if (url.pathname === "/") file = /powershell|windowspowershell/i.test(request.headers.get("user-agent") || "") ? "install.ps1" : "install.sh";
    else return new Response("not found\n", { status: 404 });
    const upstream = await fetch(RAW + file, { cf: { cacheTtl: 300, cacheEverything: true } });
    if (!upstream.ok) return new Response("the install script is not available right now\n", { status: 502 });
    return new Response(request.method === "HEAD" ? null : upstream.body, {
      headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "public, max-age=300" },
    });
  },
};
