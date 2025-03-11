def create_comparison_figure(generators_dict, full_dataset, query_class, sigma_values, hyperparams, device, output_path="comparison_figure.png"):
    from torchvision.utils import save_image, make_grid
    import torch

    classes_to_generate = [c for c in range(hyperparams["n_classes"]) if c != query_class]
    all_rows = []
    for cls in classes_to_generate:
        real_img = None
        for (img, label) in full_dataset:
            if label == cls:
                real_img = img.unsqueeze(0)
                break
        if real_img is None:
            continue

        real_img_disp = (real_img + 1) / 2  # [-1,1] -> [0,1]
        row_imgs = [real_img_disp]

        for sigma in sigma_values:
            gen = generators_dict[sigma]
            noise = torch.randn(1, hyperparams["latent_dim"], device=device)
            labels = torch.tensor([cls], device=device)
            with torch.no_grad():
                fake_img = gen(noise, labels)
            fake_img = (fake_img + 1) / 2
            row_imgs.append(fake_img.cpu())

        row_imgs_tensor = torch.cat(row_imgs, dim=0)
        row_grid = make_grid(row_imgs_tensor, nrow=len(row_imgs), padding=2, normalize=False)
        all_rows.append(row_grid)

    if len(all_rows) > 0:
        final_grid = torch.cat(all_rows, dim=1)
        save_image(final_grid, output_path)
        print(f"Comparative image saved in: {output_path}")
    else:
        print("No images found to generate the comparative figure.")