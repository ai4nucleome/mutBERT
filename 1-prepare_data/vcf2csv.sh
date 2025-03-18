for chr in {1..22} X; do
    if [ "$chr" == 'X' ]; then
        input_file="1kGP_high_coverage_Illumina.chr${chr}.filtered.SNV_INDEL_SV_phased_panel.v2.vcf.gz"
    else
        input_file="1kGP_high_coverage_Illumina.chr${chr}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    fi

    output_file="seq_npy_data/chr${chr}/clean.chr${chr}.csv"

    if [ -f "$input_file" ]; then
        echo "Processing $input_file -> $output_file"
        bcftools annotate --remove '^INFO/AN,INFO/AF,INFO/AC' "$input_file" | \
        pv | \
        bcftools filter -e 'INFO/AF<=0' | \
        bcftools filter -i 'INFO/AF > 0.02' | \
        bcftools view -v snps | \
        bcftools norm -m +any | \
        bcftools query -f '%CHROM,%POS,%REF,%ALT,%AF\n' -o "$output_file"

        # echo "finished"
    else
        echo "File $input_file not found, skipping..."
    fi
done
