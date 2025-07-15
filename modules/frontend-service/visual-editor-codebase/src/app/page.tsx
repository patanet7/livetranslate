import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function Home() {
  return (
    <div className="min-h-screen bg-bg-primary text-text-primary">
      {/* Navigation */}
      <nav className="h-14 border-b border-border-subtle flex items-center justify-between px-6">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-accent-green rounded"></div>
          <span className="text-lg font-semibold">RuleCanvas</span>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/editor">
            <Button className="bg-accent-green text-black hover:bg-accent-green/90">
              Open Editor
            </Button>
          </Link>
        </div>
      </nav>

      <main className="container mx-auto px-6 py-16 max-w-6xl">
        {/* Hero Section */}
        <div className="text-center mb-20">
          <h1 className="text-6xl font-semibold mb-6 leading-tight">
            Design rules in minutes
            <br />
            <span className="text-accent-green">Ship logic instantly</span>
          </h1>
          <p className="text-lg text-text-secondary mb-8 max-w-2xl mx-auto">
            RuleCanvas lets you craft, test, and deploy dynamic business rules with a visual node editor built for modern stacks.
          </p>
          <div className="flex items-center gap-4 justify-center mb-12">
            <Link href="/editor">
              <Button size="lg" className="bg-accent-green text-black hover:bg-accent-green/90">
                Open the editor
              </Button>
            </Link>
            <Button size="lg" variant="outline">
              Watch a 3-min demo
            </Button>
          </div>
          
          {/* Client Logos */}
          <div className="flex items-center justify-center gap-8 opacity-60">
            <div className="text-xs text-text-secondary tracking-wider uppercase">Trusted by teams at</div>
            <div className="flex items-center gap-6">
              {['Stripe', 'Shopify', 'Notion', 'Vercel', 'GitHub', 'Netlify'].map((company) => (
                <div key={company} className="text-sm text-text-secondary font-medium">{company}</div>
              ))}
            </div>
          </div>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-20">
          <Card className="bg-bg-card border-border-subtle hover:border-accent-green transition-colors">
            <CardHeader>
              <div className="w-10 h-10 bg-accent-green/10 rounded-lg flex items-center justify-center mb-4">
                <span className="text-accent-green text-xl">🎯</span>
              </div>
              <CardTitle className="text-xl">Visual Node Editor</CardTitle>
              <CardDescription className="text-text-secondary">
                Drag-and-drop interface for building complex rule flows
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-accent-green rounded-full"></div>
                  Drag-and-drop nodes
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-accent-green rounded-full"></div>
                  Zoom, pan, snap-to-grid
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-accent-green rounded-full"></div>
                  Keyboard shortcuts
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="bg-bg-card border-border-subtle hover:border-accent-green transition-colors">
            <CardHeader>
              <div className="w-10 h-10 bg-accent-green/10 rounded-lg flex items-center justify-center mb-4">
                <span className="text-accent-green text-xl">⚡</span>
              </div>
              <CardTitle className="text-xl">Real-time Validation</CardTitle>
              <CardDescription className="text-text-secondary">
                Instant feedback on rule logic and connections
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Email format</span>
                  <Badge variant="default" className="bg-green-600/20 text-green-400">Valid</Badge>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Logic flow</span>
                  <Badge variant="default" className="bg-green-600/20 text-green-400">Connected</Badge>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Syntax check</span>
                  <Badge variant="default" className="bg-yellow-600/20 text-yellow-400">Warning</Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-bg-card border-border-subtle hover:border-accent-green transition-colors">
            <CardHeader>
              <div className="w-10 h-10 bg-accent-green/10 rounded-lg flex items-center justify-center mb-4">
                <span className="text-accent-green text-xl">🚀</span>
              </div>
              <CardTitle className="text-xl">Edge Deploy</CardTitle>
              <CardDescription className="text-text-secondary">
                Deploy rules globally with one-click deployment
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-sm space-y-2">
                <div className="text-accent-green">Global edge network</div>
                <div className="text-text-secondary">Sub-50ms latency worldwide</div>
                <div className="text-text-secondary">Auto-scaling infrastructure</div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-bg-card border-border-subtle hover:border-accent-green transition-colors">
            <CardHeader>
              <div className="w-10 h-10 bg-accent-green/10 rounded-lg flex items-center justify-center mb-4">
                <span className="text-accent-green text-xl">📝</span>
              </div>
              <CardTitle className="text-xl">Version Control</CardTitle>
              <CardDescription className="text-text-secondary">
                Track changes and collaborate with your team
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-accent-green rounded-full"></div>
                  Git integration
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  Branch management
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  Rollback support
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-bg-card border-border-subtle hover:border-accent-green transition-colors">
            <CardHeader>
              <div className="w-10 h-10 bg-accent-green/10 rounded-lg flex items-center justify-center mb-4">
                <span className="text-accent-green text-xl">👥</span>
              </div>
              <CardTitle className="text-xl">Team Collaboration</CardTitle>
              <CardDescription className="text-text-secondary">
                Real-time collaboration and commenting
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="text-text-secondary">Live cursors</div>
                <div className="text-text-secondary">Comment threads</div>
                <div className="text-text-secondary">Permission management</div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-bg-card border-border-subtle hover:border-accent-green transition-colors">
            <CardHeader>
              <div className="w-10 h-10 bg-accent-green/10 rounded-lg flex items-center justify-center mb-4">
                <span className="text-accent-green text-xl">🔧</span>
              </div>
              <CardTitle className="text-xl">Language-agnostic Runtime</CardTitle>
              <CardDescription className="text-text-secondary">
                Deploy rules in any language or platform
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-bg-primary rounded p-3 text-xs font-mono">
                <div className="text-accent-green">$ npm i rulecanvas</div>
                <div className="text-text-secondary mt-1">
                  Package installed successfully
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* CTA Section */}
        <div className="text-center bg-bg-card rounded-lg border border-border-subtle p-16">
          <h2 className="text-4xl font-semibold mb-4">
            Ready to build?
          </h2>
          <p className="text-lg text-text-secondary mb-8 max-w-xl mx-auto">
            Start creating powerful rule flows in minutes. No credit card required.
          </p>
          <div className="flex items-center gap-4 justify-center">
            <Link href="/editor">
              <Button size="lg" className="bg-accent-green text-black hover:bg-accent-green/90">
                Start building now
              </Button>
            </Link>
            <Button size="lg" variant="outline">
              View templates
            </Button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border-subtle py-8 px-6 mt-20">
        <div className="container mx-auto max-w-6xl text-center">
          <p className="text-sm text-text-secondary">
            Made with <span className="text-accent-green">Orchids</span> •{" "}
            <a href="https://orchids.app" className="text-accent-green hover:text-accent-green/80 transition-colors">
              Orchids
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}