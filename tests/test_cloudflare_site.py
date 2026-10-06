from __future__ import annotations

import unittest

from deployments.cloudflare.site import API_HOSTS, PUBLIC_HOSTS, landing_page


class CloudflareSiteTests(unittest.TestCase):
    def test_public_hosts_are_explicit(self) -> None:
        self.assertEqual(PUBLIC_HOSTS, {"serpentos.dev", "www.serpentos.dev"})

    def test_api_host_is_explicit(self) -> None:
        self.assertEqual(API_HOSTS, {"api.serpentos.dev"})

    def test_landing_page_contains_product_and_api_host(self) -> None:
        page = landing_page()
        self.assertIn("<title>SerpentOS</title>", page)
        self.assertIn("api.serpentos.dev", page)
        self.assertIn("Deterministic", page)
        self.assertIn("Auditable", page)


if __name__ == "__main__":
    unittest.main()
