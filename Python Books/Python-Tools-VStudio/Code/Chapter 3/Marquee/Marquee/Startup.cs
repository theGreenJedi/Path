using Microsoft.Owin;
using Owin;

[assembly: OwinStartupAttribute(typeof(Marquee.Startup))]
namespace Marquee
{
    public partial class Startup {
        public void Configuration(IAppBuilder app) {
            ConfigureAuth(app);
        }
    }
}
