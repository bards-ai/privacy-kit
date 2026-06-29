import { PreviewTool } from "@/components/preview-tool";
import { PageHeader } from "@/components/ui";

export default function PreviewPage() {
  return (
    <>
      <PageHeader
        title="Live preview"
        description="Paste any text to see what the on-device model detects and how it would be pseudonymized. Nothing here is stored or logged."
      />
      <PreviewTool />
    </>
  );
}
