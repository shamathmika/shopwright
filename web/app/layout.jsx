export const metadata = { title: "ShopWright" };

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "system-ui, sans-serif", maxWidth: 760, margin: "40px auto", padding: "0 16px" }}>
        {children}
      </body>
    </html>
  );
}
