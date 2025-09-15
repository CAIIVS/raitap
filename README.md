<img src="images/tech_assessment_platform_logo.png" width="800">

## Purpose
The certainty pipeline is dedicated to the certification of responsible AI.
Four dimensions can be certified following MLOps practices.

The four dimensions are:
* Transparency
* Reliability
* Safety and security
* Autonomy and control


## Development Milestones
February 2023	CertAInty project is launched.  
August 2023		First documentation is created.  
January 2024	DGX Cluster migrated to new hardware.  
August 2024		DGX Cluster migrated to new software.  


### Overview over Cluster Migration Update (Summer 2024)
| Functionality         | Pre 2024          | Current Version   |
| --------              | --------          | ---               |
| Data Access   	      | Insecure Access	  | Secure Access     |
| Cluster Access    	  | SSH       	      | SSH               |
| Autostart	            | Docker	          | Podman            |
| CI/CD	                | GitHub            | GitHub            |
| Job Scheduling	      | AirFlow           | AirFlow           |
| Artifact Gathering	  | MLFlow	          | MLFlow            |
| Dataset Management	  | Oxen              | Oxen              |
| Job resubmit          | No resubmit	      | Every sunday      |


## Design Decisions
**Non-personal Account.** By having an extra account that is bound to Ricardo rather than the personal account of the developer, there are no problems with joint maintenance by several developers and by leaving staff.

**ONNX Standard.** This standard is framework-independent, working well for tensorflow and pytorch. With few adjustments to the code this standard should be attainable for every use case. Versioning is easy too.

**Hydra instead of AirFlow.** Use of Hydra keeps all programming in python language. This library is more flexible, faster, and more resource-efficient. Containers are no longer required and config files make the jobs readable and structured. While there is some initial workload to get up to speed with Hydra, there is less work during operation and further development of the tech assessment platform.

**Oxen on wait.** Dataset versioning is not as important as other aspects of the tech assessment platform. Therefore it can be integrated at a later time. This corresponds to a minimal viable product approach.
