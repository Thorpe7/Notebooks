"""
Main script to restructure TCIA data and ingest into XNAT.

This script:
1. Restructures TCIA data using metadata.csv
2. Organizes data into XNAT-compatible hierarchy using DICOM UIDs
3. Prepares data in ready_for_ingest/ directory
4. Optionally ingests the data into XNAT

Usage:
    Edit the configuration variables below, then run:
    python src/tcia_to_xnat.py

    For XNAT ingestion, set XNAT_SERVER and ensure ALIAS and SECRET
    environment variables are set with your XNAT credentials.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Import restructuring function from utils
from utils.reorg_tcia_data import restructure_tcia_data
from utils.ingest_reorg_tcia_data import ingest_tcia_to_xnat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION - Edit these variables as needed
# ============================================================================

# Path to manifest directory containing metadata.csv and DICOM data
MANIFEST_DIR = Path('data/manifest-1764020981210')

# Output directory for XNAT-ready data structure
OUTPUT_DIR = Path('data/ready_for_ingest')

# Set to True to preview changes without copying/uploading files
DRY_RUN = False

# ----------------------------------------------------------------------------
# XNAT Ingestion Settings
# ----------------------------------------------------------------------------
# Set XNAT_SERVER to enable automatic ingestion after restructuring
# Leave as None to only restructure without ingesting
XNAT_SERVER: Optional[str] = "https://tap.embarklabs.ai/"

# Optional: Override project ID (defaults to directory name from restructured data)
XNAT_PROJECT_ID: Optional[str] = None

# XNAT credentials are read from environment variables.
# Specify the names of the environment variables containing your credentials:
XNAT_USER_ENV_VAR: str = 'ALIAS'      # Environment variable for XNAT username
XNAT_PASSWORD_ENV_VAR: str = 'SECRET'  # Environment variable for XNAT password

# ============================================================================


def main():
    """Main execution function."""

    # Validate paths
    if not MANIFEST_DIR.exists():
        logger.error(f"Manifest directory does not exist: {MANIFEST_DIR}")
        return 1

    metadata_file = MANIFEST_DIR / 'metadata.csv'
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return 1

    # Display configuration
    logger.info("=" * 70)
    logger.info("TCIA to XNAT Data Restructuring")
    logger.info("=" * 70)
    logger.info(f"Manifest directory: {MANIFEST_DIR.absolute()}")
    logger.info(f"Output directory:   {OUTPUT_DIR.absolute()}")
    logger.info(f"Dry run:            {DRY_RUN}")
    logger.info("=" * 70)

    try:
        # Execute restructuring
        stats = restructure_tcia_data(
            manifest_dir=MANIFEST_DIR,
            output_dir=OUTPUT_DIR,
            dry_run=DRY_RUN
        )

        # Display results
        logger.info("")
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info(f"✓ Subjects processed:  {stats['subjects']}")
        logger.info(f"✓ Sessions created:    {stats['sessions']}")
        logger.info(f"✓ Scans organized:     {stats['scans']}")
        logger.info(f"✓ Files copied:        {stats['files_copied']}")
        logger.info("=" * 70)

        if DRY_RUN:
            logger.info("")
            logger.info("This was a DRY RUN - no files were copied.")
            logger.info("Set DRY_RUN = False in the script to perform actual restructuring.")
        else:
            logger.info("")
            logger.info(f"✓ Data is ready for XNAT ingestion at: {OUTPUT_DIR.absolute()}")

        # Perform XNAT ingestion if server is configured
        if XNAT_SERVER:
            logger.info("")
            logger.info("=" * 70)
            logger.info("XNAT INGESTION")
            logger.info("=" * 70)

            ingest_stats = ingest_tcia_to_xnat(
                data_dir=OUTPUT_DIR,
                xnat_server=XNAT_SERVER,
                project_id=XNAT_PROJECT_ID,
                dry_run=DRY_RUN,
                user_env_var=XNAT_USER_ENV_VAR,
                password_env_var=XNAT_PASSWORD_ENV_VAR
            )

            logger.info("")
            logger.info("=" * 70)
            logger.info("INGESTION SUMMARY")
            logger.info("=" * 70)
            logger.info(f"✓ Subjects ingested:  {ingest_stats['subjects']}")
            logger.info(f"✓ Sessions created:   {ingest_stats['sessions']}")
            logger.info(f"✓ Scans uploaded:     {ingest_stats['scans']}")
            if DRY_RUN:
                logger.info(f"✓ Files counted:      {ingest_stats['files']}")
            logger.info("=" * 70)
        else:
            logger.info("")
            logger.info("Next steps:")
            logger.info("  1. Review the restructured data in ready_for_ingest/")
            logger.info("  2. Set XNAT_SERVER in this script to enable automatic ingestion")
            logger.info(f"  3. Ensure {XNAT_USER_ENV_VAR} and {XNAT_PASSWORD_ENV_VAR} environment variables are set")
            logger.info("  4. Re-run to ingest data into XNAT")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during restructuring: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
