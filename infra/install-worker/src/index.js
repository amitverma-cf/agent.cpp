// agent.qozu.fyi only redirects to the two install scripts in the repository (no pages, no docs, nothing is proxied or stored here).
//   curl -fsSL https://agent.qozu.fyi/install.sh | sh        (Linux)
//   irm https://agent.qozu.fyi/install.ps1 | iex             (Windows PowerShell)
// "/" picks one by User-Agent, so the bare address works too.
const RAW = "https://raw.githubusercontent.com/amitverma-cf/agent.cpp/main/scripts/";

export default {
  fetch(request) {
    const { pathname } = new URL(request.url);
    let file;
    if (pathname === "/install.sh") file = "install.sh";
    else if (pathname === "/install.ps1") file = "install.ps1";
    else if (pathname === "/") file = /powershell/i.test(request.headers.get("user-agent") || "") ? "install.ps1" : "install.sh";
    else return new Response("not found\n", { status: 404 });
    return new Response(null, { status: 302, headers: { location: RAW + file, "cache-control": "public, max-age=300" } });
  },
};
