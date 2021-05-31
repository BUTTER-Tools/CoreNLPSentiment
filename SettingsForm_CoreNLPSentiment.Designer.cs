namespace CoreNLPSentiment
{
    partial class SettingsForm_CoreNLPSentiment
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(SettingsForm_CoreNLPSentiment));
            this.OKButton = new System.Windows.Forms.Button();
            this.UseBuiltInSentenceSplitterCheckbox = new System.Windows.Forms.CheckBox();
            this.IncludeTextCheckbox = new System.Windows.Forms.CheckBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // OKButton
            // 
            this.OKButton.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.OKButton.Location = new System.Drawing.Point(186, 238);
            this.OKButton.Name = "OKButton";
            this.OKButton.Size = new System.Drawing.Size(118, 40);
            this.OKButton.TabIndex = 6;
            this.OKButton.Text = "OK";
            this.OKButton.UseVisualStyleBackColor = true;
            this.OKButton.Click += new System.EventHandler(this.OKButton_Click);
            // 
            // UseBuiltInSentenceSplitterCheckbox
            // 
            this.UseBuiltInSentenceSplitterCheckbox.AutoSize = true;
            this.UseBuiltInSentenceSplitterCheckbox.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.UseBuiltInSentenceSplitterCheckbox.Location = new System.Drawing.Point(34, 119);
            this.UseBuiltInSentenceSplitterCheckbox.Name = "UseBuiltInSentenceSplitterCheckbox";
            this.UseBuiltInSentenceSplitterCheckbox.Size = new System.Drawing.Size(322, 20);
            this.UseBuiltInSentenceSplitterCheckbox.TabIndex = 7;
            this.UseBuiltInSentenceSplitterCheckbox.Text = "Use Built-in Sentence Tokenizer (Recommended)";
            this.UseBuiltInSentenceSplitterCheckbox.UseVisualStyleBackColor = true;
            // 
            // IncludeTextCheckbox
            // 
            this.IncludeTextCheckbox.AutoSize = true;
            this.IncludeTextCheckbox.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.IncludeTextCheckbox.Location = new System.Drawing.Point(34, 40);
            this.IncludeTextCheckbox.Name = "IncludeTextCheckbox";
            this.IncludeTextCheckbox.Size = new System.Drawing.Size(212, 20);
            this.IncludeTextCheckbox.TabIndex = 8;
            this.IncludeTextCheckbox.Text = "Include Analyzed Text in Output";
            this.IncludeTextCheckbox.UseVisualStyleBackColor = true;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(31, 65);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(360, 13);
            this.label1.TabIndex = 9;
            this.label1.Text = "This option will include each individual sentence as a unique row of output.";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(31, 142);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(417, 52);
            this.label2.TabIndex = 10;
            this.label2.Text = resources.GetString("label2.Text");
            // 
            // SettingsForm_VADER
            // 
            this.AcceptButton = this.OKButton;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(491, 290);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.IncludeTextCheckbox);
            this.Controls.Add(this.UseBuiltInSentenceSplitterCheckbox);
            this.Controls.Add(this.OKButton);
            this.DoubleBuffered = true;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "SettingsForm_VADER";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Plugin Name";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.Button OKButton;
        private System.Windows.Forms.CheckBox UseBuiltInSentenceSplitterCheckbox;
        private System.Windows.Forms.CheckBox IncludeTextCheckbox;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
    }
}